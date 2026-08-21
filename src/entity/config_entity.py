from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    kaggle_dataset_url: str
    raw_data_manifest: Path

@dataclass
class DataTrasnformationConfig:
    image_size: int
    batch_size: int
    num_workers: int

@dataclass
class Classification:
    data_transformation: DataTrasnformationConfig
    epochs: int
    patience: int