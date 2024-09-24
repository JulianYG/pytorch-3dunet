import torch
import torch.nn as nn
import numpy as np
# import torchvision.models as models
from pytorch3dunet.unet3d.buildingblocks import DoubleConv, ResNetBlock, ResNetBlockSE, \
    create_encoders, MLP, ClassificationHead
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level

import nibabel as nib

_NUM_CLASSES = 3

class ResNet3D(nn.Module):
    def __init__(self,
                in_channels: int = 1,
                out_channels: int = 32,
                kernel_size_custom: list[int] = [5, 5, 5],
                emb_size: int = 2048,
                depth: int = 8,
                n_classes: int = _NUM_CLASSES,
                # finetune: bool = False,
                **kwargs):
        super().__init__()
        raw_mask = np.expand_dims(
            np.array(nib.load('/data/seg.nii').dataobj, dtype=np.float32), axis=0).transpose((0, 2, 3, 1))
        self.batched_mask = np.expand_dims(raw_mask, axis=0)
        # self.segment_mask = torch.from_numpy(batched_mask).to('cuda')
        resnet_model = torch.hub.load('/root/.cache/torch/hub/facebookresearch_pytorchvideo_main', 'slow_r50', source='local', pretrained=False)
        state_dict = torch.load("/model/resnet_slow_50.pth")
        resnet_model.load_state_dict(state_dict["model_state"])
        # for param in resnet_model.parameters():
        #     param.requires_grad = False     # freeze resnet weights  

        # modify stem layers
        stem_conv = nn.Conv3d(
            2, 64, (7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )
        nn.init.xavier_uniform_(stem_conv.weight)
        resnet_model.blocks[0].conv = stem_conv
        # stem_conv = resnet_model.blocks[0].conv
        # stem_conv.in_channels = 2
        # stem_conv.kernel_size = (7, 7, 7)
        # stem_conv.stride = (2, 2, 2)
        # stem_conv.padding = (3, 3, 3)

        stem_pool = nn.AvgPool3d((3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        resnet_model.blocks[0].pool = stem_pool
        # stem_pool = resnet_model.blocks[0].pool
        # stem_pool.kernel_size = (3, 3, 3)
        # stem_pool.stride = (2, 2, 2)
        # stem_pool.padding = (1, 1, 1)
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
            104448, hidden_channels=[1024,256,64], 
            activation_layer=nn.LeakyReLU,
            # norm_layer=nn.BatchNorm1d, 
            # dropout=0.4,
        )
        self.relu = nn.ReLU()
        self.final_fc = nn.Linear(64, _NUM_CLASSES, bias=False)
        # self.fc = ClassificationHead(emb_size, n_classes)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # # Create mask by batch size
        # batch_size = x.shape[0]
        # segment_mask = torch.from_numpy(self.batched_mask).to('cuda')
        # mask = segment_mask.repeat(batch_size, 1, 1, 1, 1)
        # # stack the mask channel on data
        # x = torch.cat((x, mask), 1)
        # treat grayscale as RGB
        # x = self.r3d(x.repeat(1, 3, 1, 1, 1))
        x = self.r3d(x)
        x = self.final_mlp(x)
        x = self.relu(x)
        x = self.final_fc(x)
        # if not self.training:
        x = self.final_activation(x)
        return x
