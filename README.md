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
···
python main.py
···

## Results
I did not refer to another implementation of pytorch. 
In order to make it easier to compare my code with the tensorflow version, 
the code is constructed according to the tensorflow structure.<br>

The following is the result: <br>
Loss changes with epochs: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/loss.png)<br>
Acc changes with epochs: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/acc.png)<br>
Dimensionality reduction visualization of test results: <br>
![pic1](https://github.com/taishan1994/pytorch_gat/raw/master/tsne.png)<br>