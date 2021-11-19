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

# Listing modules gives a module for each Sequential group *and* a module for
# each module within that
# Iterating over model children just gives the top-level modules, e.g.
# for VGG16 get 3 (Sequential encoder stack, avgpool, classifier)

i = 0
print("\nModules:")
for module in vgg.modules():
    print("module", i, module)
    i += 1

i = 0
print("\nChildren:")
for child in vgg.children():
    print("child", i, child)
    i += 1

# https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
mod_vgg = torch.nn.Sequential(*(list(vgg.children())[:-1]))
print("\nVGG after classifier removal:")
print(mod_vgg)
# TODO check if pretrained weights retained

i = 0
print("Subchildren of encoder")
generator = mod_vgg.children()
child0 = next(mod_vgg.children())
for child in child0.children():
    print("subchild", i, child)
    i += 1
