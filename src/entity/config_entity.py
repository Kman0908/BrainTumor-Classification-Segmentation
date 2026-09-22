from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    kaggle_dataset_url: str
    raw_data_manifest: Path

@dataclass
class DataTransformationConfig:
    image_size: int
    batch_size: int
    num_workers: int

@dataclass
class Segmentation:
    data_transformation: DataTransformationConfig
    epochs: int
    patience: int

@dataclass
class Classification:
    data_transformation: DataTransformationConfig
    epochs: int
    patience: int

@dataclass 
class ModelTrainerConfig:
    num_classes: int
    learning_rate: float
    checkpoint_dir: str
    epochs: int

@dataclass
class ModelEvaluationConfig:
    checkpoint_dir: str
    num_classes: int
    threshold: float