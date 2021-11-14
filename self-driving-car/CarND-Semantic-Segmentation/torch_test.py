# Installed from pytorch channel:
# conda install pytorch torchvision -c pytorch
import torch
import torchvision.models as models


print("cuda available: ", torch.cuda.is_available())

# Downloads on Win10 by default to C:\Users\<username>/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# Source: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
vgg = models.vgg16(pretrained=True)

print("Model structure:")
print(vgg)

# Sequential: a subunit of connected layers, forward() method defined implicitly
# ModuleList: a collection of non-connected layers
# ModuleDict: a collection of layers as dict

print("\nModules:")
for module in vgg.modules():
    print(module)
