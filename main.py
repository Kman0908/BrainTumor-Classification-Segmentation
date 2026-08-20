from src.components.data_ingestion import DataIngestion
from src.entity import config_entity
from src.utils import read_yaml
from pathlib import Path

yaml_config = read_yaml('config/config.yaml')

di_conf = yaml_config['data_ingestion']

data_ingestion_config = config_entity.DataIngestionConfig(
    kaggle_dataset_url = di_conf['kaggle_dataset_url'],
    raw_data_manifest = Path(di_conf['raw_data_manifest']),
)

if __name__ == "__main__":
    data_ingestion_obj = DataIngestion(data_ingestion_config)
    manifest_path = data_ingestion_obj.initiate_ingestion()