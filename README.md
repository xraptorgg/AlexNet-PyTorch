# AlexNet implementation in PyTorch


This is an implementation of AlexNet architecture proposed by Alex Krizhevsky et al. in the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) using PyTorch.

The Jupyter Notebook contains details about the architecture and implementation steps, the Python script contains the code.

The Jupyter Notebook and Python files also contain image pipeline for the Tiny ImageNet dataset, however I did not train the model on the dataset due to hardware limitations. If you wish to train the model using the Tiny ImageNet dataset then you should download it from [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), I did not include the dataset in the repository as it is quite large, however it is very straight forward to download and train the model after you download it, just mention the file path of the `tiny-imagenet-200` folder in the `DATA_PATH` in `main.py`.


**NOTE:** This repository contains the model architecture of the original paper as proposed by Krizhevsky et al., the original architecture was trained on the ILSVRC 2012 dataset consisting of 1.2 million images distributed among 1000 classes. While the AlexNet architecture is a good model for image classification tasks, the hyperparameters such as number of activation maps, kernel size, stride, etc must be tuned according to the problem. Also during my limited time using AlexNet on CIFAR10, the Xavier Glorot initialization suggeted in the paper hindered the ability of the model to learn, you should try Kaiming He initialization if possible. Local Response Normalization also resulted in poor performance.


<div>
<img src="https://cdn.discordapp.com/attachments/418819379174572043/1079767102631723049/alexnet.png" width="1100" alt = "Alex Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks">
</div>
