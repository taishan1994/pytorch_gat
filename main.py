import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
from model import *

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = load_data("cora")
adj = data['adj']
features = data['features']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']
train_mask = data['train_mask']
val_mask = data['val_mask']
test_mask = data['test_mask']
train_my_labels = data['train_my_labels']
val_my_labels = data['val_my_labels']
test_my_labels = data['test_my_labels']
my_labels = data['my_labels']

features, spars = preprocess_features(features)

#节点数目
nb_nodes = features.shape[0]
#特征维度
ft_sizes = features.shape[1]
#类别数目
nb_classes = my_labels.shape[0]

#将邻接矩阵的稀疏形式转换为原始矩阵
adj = adj.todense()

#新增加一个维度
adj = adj[np.newaxis]
features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
#train_mask = train_mask[np.newaxis]
#val_mask = val_mask[np.newaxis]
#test_mask = test_mask[np.newaxis]

biases = torch.from_numpy(adj_to_bias(adj, [nb_nodes], nhood=1)).float().to(device)

features = torch.from_numpy(features)
#pytorch输入的特征:[batch, features，nodes]，第二位是特征维度
#而tensorflow的输入是：[batch, nodes, features]
features = torch.transpose(features,2,1).to(device)

#定义相关变量
hid_units=[8]
n_heads=[8, 1]
epochs = 2400
lr = 0.01

#定义模型
gat = GAT(nb_classes=nb_classes,
      nb_nodes=nb_nodes, 
      attn_drop=0.0, 
      ffd_drop=0.0, 
      bias_mat=biases, 
      hid_units=hid_units, 
      n_heads=n_heads, 
      residual=False).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=gat.parameters(),lr=lr,betas=(0.9, 0.99))


#y_train = torch.from_numpy(np.where(y_train==1)[2])
#y_val = torch.from_numpy(np.where(y_val==1)[2])
#y_test = torch.from_numpy(np.where(y_test==1)[2])
train_my_labels = torch.from_numpy(train_my_labels).long().to(device)
val_my_labels = torch.from_numpy(val_my_labels).long().to(device)
test_my_labels = torch.from_numpy(test_my_labels).long().to(device)

train_mask = np.where(train_mask == 1)[0]
val_mask = np.where(val_mask == 1)[0]
test_mask = np.where(test_mask == 1)[0]
train_mask = torch.from_numpy(train_mask).to(device)
val_mask = torch.from_numpy(val_mask).to(device)
test_mask = torch.from_numpy(test_mask).to(device)

print("训练节点个数：", len(train_my_labels))
print("验证节点个数：", len(val_my_labels))
print("测试节点个数：", len(test_my_labels))



def train():
  gat.train()
  correct = 0
  optimizer.zero_grad()
  outputs = gat(features)
  train_mask_outputs = torch.index_select(outputs, 0, train_mask)
  #print("train_mask_outputs.shape:",train_mask_outputs.shape)
  #print("train_my_labels.shape[0]:",train_my_labels.shape[0])
  _, preds =torch.max(train_mask_outputs.data, 1)
  loss = criterion(train_mask_outputs, train_my_labels)
  loss.backward()
  optimizer.step()
  correct += torch.sum(preds == train_my_labels).to(torch.float32)
  acc = correct / train_my_labels.shape[0]
  return loss,acc


def val():
  gat.eval()
  with torch.no_grad():
    correct = 0
    outputs = gat(features)
    val_mask_outputs = torch.index_select(outputs, 0, val_mask)
    #print("val_mask_outputs.shape:",val_mask_outputs.shape)
    #print("val_my_labels.shape[0]:",val_my_labels.shape[0])
    _, preds =torch.max(val_mask_outputs.data, 1)
    loss = criterion(val_mask_outputs, val_my_labels)
    correct += torch.sum(preds == val_my_labels).to(torch.float32)
    acc = correct / val_my_labels.shape[0]
  return loss,acc

def test():
  gat.eval()
  with torch.no_grad():
    correct = 0
    outputs = gat(features)
    test_mask_outputs = torch.index_select(outputs, 0, test_mask)
    #print("test_mask_outputs.shape:",test_mask_outputs.shape)
    #print("val_my_labels.shape[0]:",val_my_labels.shape[0])
    _, preds =torch.max(test_mask_outputs.data, 1)
    loss = criterion(test_mask_outputs, test_my_labels)
    correct += torch.sum(preds == test_my_labels).to(torch.float32)
    acc = correct / test_my_labels.shape[0]
    print("TestLoss:{:.4f},TestAcc:{:.4f}".format(loss,acc))
  return loss,acc,test_mask_outputs.cpu().numpy(),test_my_labels.cpu().numpy()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
  train_loss_history = []
  val_loss_history = []
  train_acc_history = []
  val_acc_history = []
  for epoch in range(1,epochs+1):
    train_loss,train_acc = train()
    val_loss,val_acc = val()
    print("epoch:{:03d},TrainLoss:{:.4f},TrainAcc:{:.4f},ValLoss:{:.4f},ValAcc:{:.4f}"
        .format(epoch,train_loss,train_acc,val_loss,val_acc))
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
  num_epochs = range(1, epochs + 1)
  plt.plot(num_epochs, train_loss_history, 'b--')
  plt.plot(num_epochs, val_loss_history, 'r-')
  plt.title('Training and validation Loss ')
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["train_loss", 'val_loss'])
  plt.savefig("loss.png")
  plt.close()

  plt.plot(num_epochs, train_acc_history, 'b--')
  plt.plot(num_epochs, val_acc_history, 'r-')
  plt.title('Training and validation Acc ')
  plt.xlabel("Epochs")
  plt.ylabel("Acc")
  plt.legend(['train_acc','val_acc'])
  plt.savefig("acc.png")
  plt.close()

  _, _, test_data, test_labels = test()
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000) # TSNE降维，降到2
  low_dim_embs = tsne.fit_transform(test_data)
  plt.title('tsne result')
  plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], marker='o', s=5, c=test_labels)
  plt.savefig("tsne.png")
  plt.close()

main()










