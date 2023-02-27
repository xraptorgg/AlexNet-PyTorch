"""
Implementation of original AlexNet architecture in PyTorch, from the paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.

at: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""


# importing necessary libraries

import torch
import torch.nn as nn



# AlexNet model

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels = 96, kernel_size = 11, stride = 4),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5, k = 2, alpha = 10e-4, beta = 0.75),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.LocalResponseNorm(size = 5, k = 2, alpha = 10e-4, beta = 0.75),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 256 * 6 * 6, out_features = 4096),
            nn.Dropout(p = 0.5, inplace = True),
            nn.ReLU(),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.Dropout(p = 0.5, inplace = True),
            nn.ReLU(),
            nn.Linear(in_features = 4096, out_features = num_classes)
        )

        self.init_weights()

    # xavier initialization
    def init_weights(self):
        torch.manual_seed(1234)
        for layer in self.conv_block_1:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.conv_block_2:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.conv_block_3:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.conv_block_4:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)
        for layer in self.conv_block_5:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean = 0, std = 0.01)
                nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.conv_block_2[0].bias, 1)
        nn.init.normal_(self.conv_block_4[0].bias, 1)
        nn.init.normal_(self.conv_block_5[0].bias, 1)


    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        return self.classifier(x)