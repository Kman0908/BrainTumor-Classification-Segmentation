import os
import torch
import numpy as np
import pandas as pd 
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataTrasnformationConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def transformer(config: DataTrasnformationConfig):
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees = 10),
        transforms.Resize((config.image_size + 1, config.image_size + 1)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((config.image_size + 1, config.image_size + 1)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
    ])

    return train_transform, val_transform

class CustomData(Dataset):
    def __init__(self, transform, dataframe: pd.DataFrame, encoder: LabelEncoder):
        self.transform = transform
        self.dataframe = dataframe
        self.le = encoder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        
        image = Image.open(row['image']).convert('RGB')
        label = row['label']
        label = torch.tensor(self.le.transform([label])[0], dtype = torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
    
class DataTransformation:
    def __init__(self, config: DataTrasnformationConfig, train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame, le: LabelEncoder):
        self.config = config
        self.train = train
        self.test = test
        self.val = val
        self.le = le

    def initiate_transformation(self):
        try:
            logging.info(f'Data transformation started')

            train_transformation, val_transformation = transformer(config = self.config)

            train_dataset = CustomData(train_transformation, self.train, self.le)
            test_dataset = CustomData(val_transformation, self.test, self.le)
            val_dataset = CustomData(val_transformation, self.val, self.le)

            train_loader = DataLoader(train_dataset, batch_size = self.config.batch_size, shuffle = True, num_workers = self.config.num_workers)
            test_loader = DataLoader(test_dataset, batch_size = self.config.batch_size, shuffle = True, num_workers = self.config.num_workers)
            val_loader = DataLoader(val_dataset, batch_size = self.config.batch_size, shuffle = True, num_workers = self.config.num_workers)

            logging.info(f'Data Transformation completed')

            return train_loader, test_loader, val_loader
        except Exception as e:
            logging.exception(f'Error occurred at DataTransformation.initiate_transformation')
            raise CustomException(e)