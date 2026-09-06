import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from src.logger import logging
from src.exception import CustomException

class Model(nn.Module):
    def __init__(self, encoder_name: str = 'resnet50', freeze_layers: bool = True):
        super().__init__()
        try:
            self.model  = smp.UnetPlusPlus(
                encoder_name = encoder_name,
                encoder_weights = 'imagenet',
                in_channels = 3,
                classes = 1,
                activation = None
            )
            if freeze_layers:
                for param in self.model.encoder.parameters():
                    param.requires_grad = True

        except Exception as e:
            logging.exception('Error occurred at Model.__init__')
            raise CustomException(e)
    
    def freeze(self):
        try:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            
            for name, param in self.model.encoder.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True
        except Exception as e:
            logging.exception('Error occurred at Model.freeze')
            raise CustomException(e)

    def forward(self, x):
        return self.model(x)