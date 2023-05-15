import torch
from unet import UNetLite,UNet
from torchsummary import summary
model = UNetLite(n_channels=1, n_classes=2, bilinear=False)
model_normal = UNet(n_channels=1, n_classes=2, bilinear=False)
summary(model, (1, 128, 128))
summary(model_normal, (1, 128, 128))
