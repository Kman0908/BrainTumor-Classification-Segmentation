import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def transformer(config: DataTransformationConfig):
    train_transform = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.ColorJitter(brightness = 0.1, contrast = 0.5),
        A.Affine(translate_percent=(-0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
        A.HorizontalFlip(p = 0.5),
        A.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD),
        ToTensorV2(),
    ])

    return train_transform, val_transform

class CustomData(Dataset):
    def __init__(self, transformer, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.transformer = transformer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image = np.array(Image.open(row['image']).convert('RGB'))
        mask = Image.open(row['mask']).convert('L')
        mask_array = (np.array(mask) > 0).astype(np.float32)

        if self.transformer:
            augmented = self.transformer(image = image, mask = mask_array)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame):
        self.config = config
        self.train = train
        self.test = test
        self.val = val
    
    def initiate_transformation(self):
        try:
            logging.info('Data Transformation Started')

            train_transform, val_transform = transformer(config = self.config)

            train_dataset = CustomData(train_transform, self.train)
            test_dataset = CustomData(val_transform, self.test)
            val_dataset = CustomData(val_transform, self.val)

            train_loader = DataLoader(dataset = train_dataset, batch_size = 6, shuffle = True, num_workers = self.config.num_workers)
            test_loader = DataLoader(dataset = test_dataset, batch_size = 6, shuffle = False, num_workers = self.config.num_workers)
            val_loader = DataLoader(dataset = val_dataset, batch_size = 6, shuffle = False, num_workers = self.config.num_workers)

            logging.info('Transformation Completed')

            return train_loader, test_loader, val_loader
        except Exception as e:
            logging.exception('Error occurred at data_transformation.initiate_transformation')
            raise CustomException(e)


