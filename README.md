# pytorch_gat
Pytorch implementation of graph attention network:<br>
Paper address:Graph Attention Networks (Veličković et al., ICLR 2018): https://arxiv.org/abs/1710.10903<br>

The implementation is based on the official code of the graph attention network. 
Not to get the same performance as the original code, but to deepen the understanding of tensorflow and pytorch.
If you want better performance, you can refer to:<br>

Official implementation: https://github.com/PetarV-/GAT<br>
Another pytorch implementation: https://github.com/Diego999/pyGAT<br>
keras implementation：https://github.com/danielegrattarola/keras-gat<br>

You can learn how to convert tensorflow code to pytorch code from here: <br>
https://i.cnblogs.com/posts/edit;postId=13659274

## Introduction
utils.py: Read data and data processing.<br>
layer.py: Attention layer.<br>
model.py: Graph attention model network.<br>
main.py: Training, validation and testing.<br>
You can run it through：
```python
python main.py
```

## Results
I did not refer to another implementation of pytorch. 
In order to make it easier to compare my code with the tensorflow version, 
the code is constructed according to the tensorflow structure.<br>

The following is the result of running the official code:
```python
Dataset: cora
----- Opt. hyperparams -----
lr: 0.005
l2_coef: 0.0005
----- Archi. hyperparams -----
nb. layers: 1
nb. units per layer: [8]
nb. attention heads: [8, 1]
residual: False
nonlinearity: <function elu at 0x7f1b7507af28>
model: <class 'models.gat.GAT'>
(2708, 2708)
(2708, 1433)
epoch:  1
Training: loss = 1.94574, acc = 0.14286 | Val: loss = 1.93655, acc = 0.13600
epoch:  2
Training: loss = 1.94598, acc = 0.15714 | Val: loss = 1.93377, acc = 0.14800
epoch:  3
Training: loss = 1.94945, acc = 0.14286 | Val: loss = 1.93257, acc = 0.19600
epoch:  4
Training: loss = 1.93438, acc = 0.24286 | Val: loss = 1.93172, acc = 0.22800
epoch:  5
Training: loss = 1.93199, acc = 0.17143 | Val: loss = 1.93013, acc = 0.36400
。。。。。。
epoch:  674
Training: loss = 1.23833, acc = 0.49286 | Val: loss = 1.01357, acc = 0.81200
Early stop! Min loss:  1.010906457901001 , Max accuracy:  0.8219999074935913
Early stop model validation loss:  1.3742048740386963 , accuracy:  0.8219999074935913
Test loss: 1.3630210161209106 ; Test accuracy: 0.8219999074935913
```
The following is the result of running my code:
```python
(2708, 2708)
(2708, 1433)
训练节点个数： 140
验证节点个数： 500
测试节点个数： 1000
epoch:001,TrainLoss:7.9040,TrainAcc:0.0000,ValLoss:7.9040,ValAcc:0.0000
epoch:002,TrainLoss:7.9040,TrainAcc:0.0000,ValLoss:7.9039,ValAcc:0.1920
epoch:003,TrainLoss:7.9039,TrainAcc:0.0714,ValLoss:7.9039,ValAcc:0.1600
epoch:004,TrainLoss:7.9038,TrainAcc:0.1000,ValLoss:7.9039,ValAcc:0.1020
。。。。。。
epoch:2396,TrainLoss:7.0191,TrainAcc:0.8929,ValLoss:7.4967,ValAcc:0.7440
epoch:2397,TrainLoss:7.0400,TrainAcc:0.8786,ValLoss:7.4969,ValAcc:0.7580
epoch:2398,TrainLoss:7.0188,TrainAcc:0.8929,ValLoss:7.4974,ValAcc:0.7580
epoch:2399,TrainLoss:7.0045,TrainAcc:0.9071,ValLoss:7.4983,ValAcc:0.7620
epoch:2400,TrainLoss:7.0402,TrainAcc:0.8714,ValLoss:7.4994,ValAcc:0.7620
TestLoss:7.4805,TestAcc:0.7700
```

The following is the result: <br>
Loss changes with epochs: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/loss.png)<br>
Acc changes with epochs: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/acc.png)<br>
Dimensionality reduction visualization of test results: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/tsne.png)<br>