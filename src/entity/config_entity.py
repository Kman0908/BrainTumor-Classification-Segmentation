from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    kaggle_dataset_url: str
    raw_data_manifest: Path