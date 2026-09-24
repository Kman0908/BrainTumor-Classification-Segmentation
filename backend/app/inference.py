from pathlib import Path

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

from src.components.classification.model import Model as ClassificationModel
from src.components.segmentation.model import Model as SegmentationModel

from src.exception import CustomException

project_root = Path(__file__).resolve().parents[2]

classification_model_path = project_root / 'artifacts' / 'objects' / 'classification.pth'
segmentation_model_path = project_root / 'artifacts' / 'objects' / 'segmentation.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_name = [    
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

segmentation_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

class InferenceEngine:
    def __init__(self):
        self.device = device

        self.classification_model = self.load_classification_model()
        self.segmentation_model = self.load_segmentation_model()

    def load_classification_model(self):
        try:
            model = ClassificationModel(num_classes = 4, freeze_layers = False)
            state_dict = torch.load(classification_model_path, map_location = self.device, weights_only = True)

            model.load_state_dict(state_dict)
            model.to(device = self.device)
            model.eval()

            return model
        except Exception as e:
            raise CustomException(f'Failed to load Classification model')
        
    def load_segmentation_model(self):
        try:
            model = SegmentationModel(encoder_name = 'resnet50', freeze_layers = False)
            state_dict = torch.load(segmentation_model_path, map_location = self.device, weights_only = True)

            model.load_state_dict(state_dict)
            model.to(device = self.device)
            model.eval()

            return model
        except Exception as e:
            raise CustomException(f'Failed to load Segmentation model')

    def classify(self, image: Image.Image):
        try:
            image_ = classification_transform(image)
            image_ = image_.unsqueeze(0)
            image_ = image_.to(self.device)

            with torch.inference_mode():

                output = self.classification_model(image_)
                probability = torch.softmax(output, dim = 1)

                confidence, predicted = torch.max(probability, dim = 1)

            return {'class': class_name[predicted.item()],
                    'confidence': confidence.item()}
        except Exception as e:
            raise CustomException(f'Classification inference failed')

    def segment(self, image: Image.Image):
        try:
            image_tensor = segmentation_transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            with torch.inference_mode():
                output = self.segmentation_model(image_tensor)

                probability = torch.sigmoid(output)
                mask = probability > 0.5

            mask = mask.squeeze().cpu().numpy().astype('uint8') * 225
            return Image.fromarray(mask)
        except Exception as e:
            raise CustomException(f'Segmentation infernece failed')
        
