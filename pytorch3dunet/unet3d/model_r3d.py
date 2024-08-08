import torch
import torch.nn as nn
import numpy as np
# import torchvision.models as models
from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_encoders, MLP, ClassificationHead
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level

_NUM_CLASSES = 3

class ResNet3D(nn.Module):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 32,
                kernel_size_custom: list[int] = [5, 5, 5],
                emb_size: int = 2048,
                depth: int = 8,
                n_classes: int = _NUM_CLASSES,
                **kwargs):
        super().__init__()
        resnet_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        for param in resnet_model.parameters():
            param.requires_grad = False     # freeze resnet weights
        
        # modify stem layers
        stem_conv = resnet_model.blocks[0].conv
        stem_conv.kernel_size = (7, 7, 7)
        stem_conv.stride = (2, 2, 2)
        stem_conv.padding = (3, 3, 3)
        stem_pool = resnet_model.blocks[0].pool
        stem_pool.kernel_size = (3, 3, 3)
        stem_pool.stride = (2, 2, 2)
        stem_pool.padding = (1, 1, 1)
        torch.nn.init.xavier_uniform(stem_conv.weight)
        # Weights of BN layers are typically set as 1, bias as 0
        resnet_model.blocks[0].norm.weight.data.fill_(1.0)
        resnet_model.blocks[0].norm.bias.data.fill_(0)
        
        # ignore last two layers in resnet basic head, to use our own MLP and FC
        resnet_model.blocks[5].proj = nn.Identity()
        resnet_model.blocks[5].output_pool = nn.Identity()
        # Unfreeze parameters in BasicStem
        for param in resnet_model.blocks[0].parameters():
            param.requires_grad = True
        
        self.r3d = resnet_model
        self.final_mlp = MLP(
            106496, hidden_channels=[1024,256,128], 
            # norm_layer=nn.BatchNorm1d, 
            dropout=0.2)
        
        self.final_fc = nn.Linear(128, _NUM_CLASSES)
        # self.fc = ClassificationHead(emb_size, n_classes)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # treat grayscale as RGB
        x = self.r3d(x.repeat(1, 3, 1, 1, 1))
        # print(x.shape)
        x = self.final_mlp(x)
        return self.final_fc(x)



