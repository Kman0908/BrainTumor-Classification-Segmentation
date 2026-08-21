import pandas as pd
from pathlib import Path
from src.utils import read_yaml
from src.entity import config_entity
from sklearn.preprocessing import LabelEncoder
from src.components.data_ingestion import DataIngestion
from src.components.classification.data_transformation import DataTransformation

le = LabelEncoder()

yaml_config = read_yaml('config/config.yaml')

di_conf = yaml_config['data_ingestion']
data_ingestion_config = config_entity.DataIngestionConfig(
    kaggle_dataset_url = di_conf['kaggle_dataset_url'],
    raw_data_manifest = Path(di_conf['raw_data_manifest']),
)

classification = yaml_config['classification']
data_transformation_config = config_entity.DataTrasnformationConfig(
    image_size = classification['image_size'],
    batch_size = classification['batch_size'],
    num_workers = classification['num_workers']
)


if __name__ == "__main__":
    # data ingestion
    data_ingestion_obj = DataIngestion(data_ingestion_config)
    manifest_path = data_ingestion_obj.initiate_ingestion()

    # data transformation
    train = manifest_path / 'classification' / 'train.csv'
    test = manifest_path / 'classification' / 'test.csv'
    val = manifest_path / 'classification' / 'val.csv'

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test)
    val_df = pd.read_csv(val)

    le.fit(train_df['label'])
    
    data_transformation_obj = DataTransformation(data_transformation_config, train_df, test_df, val_df, le)
    classification_train, classification_test, classification_val = data_transformation_obj.initiate_transformation()