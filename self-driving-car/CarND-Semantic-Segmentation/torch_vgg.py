# Implement FCN based on VGG16 in same structure as original project,
# but using PyTorch library

import torch
import torchvision.models

# Subclass Torchvision library model to build own model with required structure
# Source code here:
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
# Nice very general FCN implementation example here:
# https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
class VggFcn(torchvision.models.VGG):
    def __init__(self, keep_prob=0.5, num_classes=2):
        self = torchvision.models.vgg16(pretrained=True) # can I do this?

        # The library model is divided into these modules as class attributes:
        #   self.features -- encoder
        #   self.avgpool -- to force 7x7 size feeding into classifier? So we can lose this?
        #   <then flattens>
        #   self.classifier -- which we don't want

        del self.avgpool
        del self.classifier


        # Construct new layers for decoder part in line with original project here
        drop_prob = 1.0 - keep_prob

        self.layer6_conv = torch.nn.Conv2d(512,        # input channels
                                    4096,       # output channels
                                    7,          # 7x7 patch from original Udacity model
                                    stride=(1,1))

        self.layer6_conv_activation = torch.nn.ReLU()

        self.layer6_dropout = torch.nn.Dropout(p=drop_prob)

        self.layer7_conv = torch.nn.Conv2d(4096,
                                            4096,
                                            1,         # 1x1 patch from original Udacity model
                                            stride=(1,1))

        self.layer7_conv_activation = torch.nn.ReLU()

        self.layer7_dropout = torch.nn.Dropout(p=drop_prob)

        # We should now have the same structure as the original Udacity version of VGG16,
        # but still need to add the decoder and skip connections as before

        # Upsample by 2. We need to work our way down from a kernel depth of 4096
        # to just our number of classes (i.e. 2). Should we do this all in one go?
        # Or keep more depth in as we work upwards? For now doing it all in one hit.
        self.layer8 = torch.nn.ConvTranspose2d(4096, # in_channels
                                               num_classes, #out_channels filters, 
                                                4, # kernel size taken from classroom example, might experiment
                                                stride=2) # stride causes upsampling

        self.layer8_convt_activation = torch.nn.ReLU()

        # Squash layer4 output with 1x1 convolution so that it has compatible filter depth (i.e. num_classes)
        self.layer4_squashed = torch.nn.Conv2d(512, # in_channels
                                                num_classes, # out_channels (new number of filters)
                                                1)    # 1x1 convolution so kernel size 1

        self.layer4_squashed_activation = torch.nn.ReLU()

        # upsample by 2
        self.layer9 = torch.nn.ConvTranspose2d(num_classes,
                                                num_classes, # filters
                                                    4, # kernel size taken from classroom example
                                                    stride=(2,2)) # stride causes upsampling

        self.layer9_convt_activation = torch.nn.ReLU()

        # Now we're at 20x72x2 so same pixel resolution as layer3_out, but need to squash that from
        # 256 filters to 2 (num_classes) before we can add it in as skip connection
        self.layer3_squashed = torch.nn.Conv2d(256, # in_channels
                                                   num_classes, # new number of filters
                                                    1)    # 1x1 convolution so kernel size 1

        self.layer3_squashed_activation = torch.nn.ReLU()

        # upsample by 8 to get back to original image size
        self.layer10 = torch.nn.ConvTranspose2d(num_classes,
                                                   num_classes,
                                                    32, # Finding quite large kernel works nicely
                                                    stride=(8,8)) # stride causes upsampling
        
        self.layer10_convt_activation = torch.nn.ReLU()

        # so now we should be at 160x576x2, same as original image size, 2 classes


    def forward(self, x):

        # Module instances callable via __callable__ class method

        # Index of encoder children in pretrained VGG16 we need to attach
        # skip layers or output to (layers don't have names, sadly): 
        #  16 MaxPool2d layer3_out
        #  23 MaxPool2d layer4_out
        # Is there a better way to access specific modules?
        i = 0
        for module in self.features.modules():
            x = module(x)
            if (i == 16) : layer3_out = x
            if (i == 23) : layer4_out = x
            i += 1

        x = self.features(x)
        x = self.layer6_conv(x)
        x = self.layer6_conv_activation(x)
        x = self.layer6_dropout(x)
        x = self.layer7_conv(x)
        x = self.layer7_conv_activation(x)
        x = self.layer7_dropout(x)
        x = self.layer8(x)
        x = self.layer8_convt_activation(x)
        x = x + layer4_out # skip layer addition
        x = self.layer9(x)
        x = self.layer9_convt_activation(x)
        x = x + layer3_out # skip layer addition
        x = self.layer10(x)
        x = self.layer10_convt_activation(x) # output predictions

        return x
