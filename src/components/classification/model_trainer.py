import gc
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.logger import logging
from src.exception import CustomException
from src.components.classification.model import Model
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_path = os.path.join(self.config.checkpoint_dir, 'classification.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Training using device: {self.device}')

    def _compute_class_weights(self):
        dataset = self.train_loader.dataset
        labels_str = dataset.dataframe['label'].values
        labels_enc = dataset.le.transform(labels_str)

        classes = np.unique(labels_enc)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels_enc)

        logging.info(f'Computed class weights: {dict(zip(dataset.le.inverse_transform(classes), weights))}')
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train(self):
        try:
            logging.info(f'Training Started')
            model = Model(num_classes = self.config.num_classes, freeze_layers = True).to(self.device)
            optim = Adam([
                {'params': [p for n, p in model.named_parameters() if 'layer3' in n], 'lr': 1e-6},
                {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': 1e-5},
                {'params': [p for n, p in model.named_parameters() if 'fc' in n], 'lr': self.config.learning_rate}
            ])

            class_weights = self._compute_class_weights()
            criterian = nn.CrossEntropyLoss(weight=class_weights)

            best_val_score = 0.0
            counter = 0
            patience = 5

            for epoch in range(self.config.epochs):
                model.train()
                total_loss = 0.0

                for image, label in self.train_loader:
                    image, label = image.to(self.device), label.to(self.device)

                    optim.zero_grad()
                    output = model(image)

                    loss = criterian(output, label)
                    loss.backward()

                    optim.step()
                    total_loss += loss.item()

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for image, label in self.val_loader:
                        image, label = image.to(self.device), label.to(self.device)

                        output = model(image)
                        pred = output.argmax(dim = 1)

                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(label.cpu().numpy())

                val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                val_macro_f1 = f1_score(all_labels, all_preds, average='macro')

                logging.info(f'Epoch: {epoch + 1} | Loss: {(total_loss / len(self.train_loader)):.4f} | '
                             f'Validation accuracy: {val_acc:.4f} | Validation macro F1: {val_macro_f1:.4f}')

                if val_macro_f1 > best_val_score:
                    best_val_score = val_macro_f1
                    counter = 0
                    torch.save(model.state_dict(), self.save_path)
                    logging.info(f'Saved best model (macro F1): {best_val_score:.4f}')
                else:
                    counter += 1
                    if counter >= patience:
                        logging.info(f'No visible improvement found after epochs: {patience}')
                        break

            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logging.exception(f'Error occurred at model_trainer.train')
            raise CustomException(e)