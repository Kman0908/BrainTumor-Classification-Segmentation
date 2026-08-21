import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def classification_manifest(self, path: Path) -> pd.DataFrame:
        try:
            logging.info('Creating Train, Test, and Validation for classification task')

            train_labels = []
            train_images = []

            root_dir = path / 'classification_task' / 'train'
            for label in os.listdir(root_dir):
                class_dir = root_dir / label
                for image in os.listdir(class_dir):
                    train_labels.append(label)
                    train_images.append(str(class_dir / image))

            train = pd.DataFrame(zip(train_images, train_labels), columns = ['image', 'label'])

            test_labels = []
            test_images = []
            
            root_dir = path / 'classification_task' / 'test'
            for label in os.listdir(root_dir):
                class_dir = root_dir / label
                for image in os.listdir(class_dir):
                    test_labels.append(label)
                    test_images.append(str(class_dir / image))

            test = pd.DataFrame(zip(test_images, test_labels), columns = ['image', 'label'])

            train, validation = train_test_split(train, test_size = 0.3, shuffle = True, stratify = train['label'], random_state = 42)
            validation = validation.reset_index(drop = True)
            train = train.reset_index(drop = True)

            logging.info(f'Created train dataframe. \nDemo:\n{train.head()}\nShape:{train.shape}')
            logging.info(f'Created test dataframe. \nDemo:\n{test.head()}\nShape:{test.shape}')
            logging.info(f'Created validation dataframe. \nDemo:\n{validation.head()}\nShape:{validation.shape}')

            return train, test, validation
        
        except Exception as e:
            logging.exception(f'Error occurred at DataIngestion.classification_manifest')
            raise CustomException(e)
        
    def segmentation_manifest(self, path: Path) -> pd.DataFrame:
        try:
            logging.info('Creating Train, Test, and Validation for segmentation task')

            train_images = []
            train_mask = []
            root_dir = path / 'segmentation_task' / 'train'

            for image in os.listdir(f'{root_dir}/images'):
                mask = image.replace('.jpg', '.png')
                mask_dir = root_dir / 'masks' / mask
                if os.path.exists(mask_dir):
                    train_images.append(str(root_dir / 'images' / image))
                    train_mask.append(mask_dir)

            train = pd.DataFrame(zip(train_images, train_mask), columns = ['image', 'mask'])

            test_mask = []
            test_images = []
            root_dir = path / 'segmentation_task' / 'test'

            for image in os.listdir(f'{root_dir}/images'):
                mask = image.replace('.jpg', '.png')
                mask_dir = root_dir / 'masks' / mask
                if os.path.exists(mask_dir):
                    test_images.append(root_dir / 'images' / image)
                    test_mask.append(mask_dir)

            test = pd.DataFrame(zip(test_images, test_mask), columns = ['image', 'mask'])

            train, validation = train_test_split(train, test_size = 0.3, shuffle = True, random_state = 42)
            validation = validation.reset_index(drop = True)
            train = train.reset_index(drop = True)

            logging.info(f'Created train dataframe. \nDemo:\n{train.head()}\nShape:{train.shape}')
            logging.info(f'Created test dataframe. \nDemo:\n{test.head()}\nShape:{test.shape}')
            logging.info(f'Created validation dataframe. \nDemo:\n{validation.head()}\nShape:{validation.shape}')

            return train, test, validation
        
        except Exception as e:
            logging.exception(f'Error occurred at DataIngestion.segmentation_manifest')
            raise CustomException(e)


    def initiate_ingestion(self) -> Path:
        try:
            path = Path(f'{os.getcwd()}/Data')
            logging.info(f'Got the data path: {path}')

            logging.info(f'Creating data manifest for classification and segmentation')
            class_train, class_test, class_val = self.classification_manifest(path)
            seg_train, seg_test, seg_val = self.segmentation_manifest(path)

            dataframes = {
                'classification': {
                    'train': class_train,
                    'test': class_test,
                    'val': class_val
                },
                'segmentation': {
                    'train': seg_train,
                    'test': seg_test,
                    'val': seg_val
                }
            }

            save_path = Path(os.getcwd()) / 'artifacts' / 'manifests'
            os.makedirs(save_path, exist_ok = True)

            for task, split in dataframes.items():
                task_dir = os.path.join(save_path, task)
                os.makedirs(task_dir, exist_ok=True)

                for type, file in split.items():
                    file_name = f'{type}.csv'
                    save = os.path.join(task_dir, file_name)

                    file.to_csv(save, index=False)
                    logging.info(f'File: {file_name}, saved at: {save}')

            return save_path
        except Exception as e:
            logging.exception(f'Error occurred at DataIngestion.create_manifest')
            raise CustomException(e)