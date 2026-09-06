import gc
import os
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import segmentation_models_pytorch as smp

from src.logger import logging
from src.exception import CustomException
from src.components.segmentation.model import Model
from src.entity.config_entity import ModelTrainerConfig

criterion_bce = nn.BCEWithLogitsLoss()
criterion_tvc = smp.losses.TverskyLoss(mode = 'binary', alpha = 0.3, beta = 0.7)

def dice_coefficient(pred, target, epsilon = 1e-6, threshold = 0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim = (1, 2, 3))
    union = pred.sum(dim = (1, 2, 3)) + target.sum(dim = (1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

def IoU(pred, target, epsilon = 1e-6, threshold = 0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim = (1, 2, 3))
    union = pred.sum(dim = (1, 2, 3)) + target.sum(dim = (1, 2, 3)) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()

def combined_loss(pred, true):
    return criterion_bce(pred, true) + criterion_tvc(pred, true)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_path = os.path.join(self.config.checkpoint_dir, 'segmentaion.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Training using device: {self.device}')
    
    def train(self):
        try:
            logging.info('Training started')
            model = Model(freeze_layers = True).to(self.device)
            optim = Adam([
                {'params': [p for n, p in model.model.encoder.named_parameters() if 'layer3' in n], 'lr': 1e-6},
                {'params': [p for n, p in model.model.encoder.named_parameters() if 'layer4' in n], 'lr': 1e-5},
                {'params': [p for p in model.model.decoder.parameters()], 'lr': 1e-4},
                {'params': [p for p in model.model.segmentation_head.parameters()], 'lr': 1e-3}
            ])
            scalar = torch.amp.GradScaler()   

            best_val_dice = 0.0
            counter = 0
            patience = 5

            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0.0

                for image, mask in self.train_loader:
                    image, mask = image.to(self.device), mask.to(self.device)

                    optim.zero_grad()

                    with torch.amp.autocast(device_type = 'cuda', dtype = torch.float16):
                        output = model(image)
                        loss = combined_loss(output, mask)
                    
                    scalar.scale(loss).backward()
                    scalar.step(optim)
                    scalar.update()

                    total_loss += loss.item()

                val_dice_scores = []
                model.eval()
                with torch.no_grad():
                    for image, mask in self.val_loader:
                        image, mask = image.to(self.device), mask.to(self.device)

                        with torch.amp.autocast(device_type = 'cuda', dtype = torch.float16):
                            output = model(image)
                        
                        dice = dice_coefficient(output, mask)
                        val_dice_scores.append(dice)
                    
                avg_val_dice = sum(val_dice_scores) / len(val_dice_scores)
                logging.info(f'Epoch: {epoch + 1} | Loss: {(total_loss / len(total_loss)):.4f} | Validation dice: {avg_val_dice:.4f}')
                
                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    counter = 0
                    torch.save(model.state_dict(), self.save_path)
                    logging.info(f'Best model with score: {best_val_dice:.4f} at {self.save_path}')
                else:
                    counter += 1
                    if counter > patience:
                        logging.info(f'No improvement found at epoch: {epoch + 1}, Early stopping triggred')
                        break
        except Exception as e:
            logging.exception('Error occurred at ModelTrainer.train')
            raise CustomException(e)

gc.collect()
torch.cuda.empty_cache()