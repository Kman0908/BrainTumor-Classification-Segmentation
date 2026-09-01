import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from src.logger import logging
from src.exception import CustomException

class Model(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_layers: bool = True):
        super().__init__()

        try:
            self.layer = resnet50(weights = ResNet50_Weights.DEFAULT)

            if freeze_layers:
                for param in self.layer.parameters():
                    param.requires_grad = False

            self.layer.fc = nn.Sequential(
                nn.Dropout(p = 0.3),
                nn.Linear(in_features = self.layer.fc.in_features, out_features = num_classes)
            )
        except Exception as e:
            logging.exception(f'Error occurred in Model.__init__')
            raise CustomException(e)

    def unfreeze(self):
        try:
            for name, param in self.layer.named_parameters():
                if name in 'layer4' or name in 'layer3':
                    param.requires_grad = True
        except Exception as e:
            logging.exception(f'Error occurred in Model.unfreeze')
            raise CustomException(e)

    def forward(self, x):
        return self.layer(x)