import os
import gc
import torch
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from src.components.classification.model import Model
from src.entity.config_entity import ModelEvaluatonConfig
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModelEvaluation:
    def __init__(self, config: ModelEvaluatonConfig, test_loader, class_map):
        self.config = config
        self.test_loader = test_loader
        self.class_label = class_map
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Testing using device: {self.device}')

    def load_model(self) -> Model:
        try:
            logging.info(f'Loading model')
            checkpoint_dir = Path(self.config.checkpoint_dir)

            if not checkpoint_dir.exists():
                raise FileNotFoundError(f'No Checkpoint found at {checkpoint_dir}')

            model = Model(self.config.num_classes, True)
            model.load_state_dict(torch.load(checkpoint_dir, map_location = self.device))
            model.to(self.device)

            logging.info(f'Model loaded')
            return model
        except Exception as e:
            logging.exception(f'Error occurred at model_evaluation.load_model')
            raise CustomException(e)

    def evaluate(self):
        try:
            model = self.load_model()
            test_pred = []
            test_label = []

            model.eval()
            with torch.no_grad():
                for image, label in self.test_loader:
                    image, label = image.to(self.device), label.to(self.device)

                    output = model(image)
                    pred = output.argmax(dim = 1)

                    test_pred.extend(pred.cpu().numpy())
                    test_label.extend(label.cpu().numpy())

            test_score = accuracy_score(test_label, test_pred)
            logging.info(f'Test accuracy score: {test_score}')
            logging.info(f'Confusion Matrix: \n{confusion_matrix(test_label, test_pred)}')
            logging.info(f'Classification Report: \n{classification_report(test_label, test_pred, labels = list(range(len(self.class_label))), target_names = self.class_label)}')

            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            logging.info(f'Error occurred at model_evaluation.evaluate')
            raise CustomException(e)