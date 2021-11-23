# Implement FCN based on VGG16 in same structure as original project,
# but using PyTorch library

import torch
import torchvision.models

# Subclass Torchvision library model to build own model with required structure
# Source code here:
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
class VggFcn(torchvision.models.vgg16):
    def __init__(self, keep_prob=0.5):
        super().__init__(pretrained=True)

        # The library model is divided into these modules as class attributes:
        #   self.features -- encoder
        #   self.avgpool -- to force 7x7 size feeding into classifier? So we can lose this?
        #   <then flattens>
        #   self.classifier -- which we don't want

        del self.avgpool
        del self.classifier

        # Index of encoder children in pretrained VGG16 we need to attach
        # skip layers or output to (layers don't have names, sadly): 
        #  16 MaxPool2d layer3_out
        #  23 MaxPool2d layer4_out
        #  30 MaxPool2d layer7_out
        # Is there a better way to access specific modules?
        i = 0
        for module in self.features.modules():
            if (i == 16) : self.layer3_out = module
            if (i == 23) : self.layer4_out = module
            if (i == 30) : self.layer7_out = module
            i += 1

        # Construct new layers for decoder part in line with original project here
        drop_prob = 1.0 - keep_prob

        self.layer6_conv = torch.nn.Conv2d(512,        # input channels
                                    4096,       # output channels
                                    7,          # 7x7 patch from original Udacity model
                                    stride=(1,1),
                                    padding='same')

        self.layer6_conv_activation = torch.nn.ReLU()

        self.layer6_dropout = torch.nn.Dropout(p=drop_prob)


    def forward(self, x):

        # Module instances callable via __callable__ class method

        x = self.features(x)
        x = self.layer6_conv(x)
        x = self.layer6_conv_activation(x)
        x = self.layer6_dropout()

        # ...skip layer additions like this
        # x = x + self.layer3_out

        return x
