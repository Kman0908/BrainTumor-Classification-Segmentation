import pandas as pd
from pathlib import Path
from src.utils import read_yaml
from src.entity import config_entity
from sklearn.preprocessing import LabelEncoder
from src.components.data_ingestion import DataIngestion
from src.components.classification.data_transformation import DataTransformation
from src.components.classification.model_trainer import ModelTrainer
from src.components.classification.model_evaluation import ModelEvaluation

from src.components.segmentation.data_transformation import DataTransformation as Transformer
from src.components.segmentation.model_trainer import ModelTrainer as Trainer
from src.components.segmentation.model_evaluation import ModelEvaluation as Evaluator

le = LabelEncoder()

yaml_config = read_yaml('config/config.yaml')

di_conf = yaml_config['data_ingestion']
data_ingestion_config = config_entity.DataIngestionConfig(
    kaggle_dataset_url = di_conf['kaggle_dataset_url'],
    raw_data_manifest = Path(di_conf['raw_data_manifest']),
)

classification = yaml_config['classification']
classification_config = config_entity.DataTransformationConfig(
    image_size = classification['image_size'],
    batch_size = classification['batch_size'],
    num_workers = classification['num_workers']
)

mt_conf = yaml_config['model_trainer']
model_trainer_config = config_entity.ModelTrainerConfig(
    num_classes = mt_conf['num_classes'],
    learning_rate = mt_conf['learning_rate'],
    checkpoint_dir = mt_conf['checkpoint_dir'],
    epochs = mt_conf['epoch']
)

me_conf = yaml_config['model_evaluation']
evlaution = config_entity.ModelEvaluatonConfig(
    checkpoint_dir = me_conf['checkpoint_dir'],
    num_classes = me_conf['num_classes']
)

segmentation = yaml_config['segmentation']
segmentation_config = config_entity.DataTransformationConfig(
    image_size = segmentation['image_size'],
    batch_size = segmentation['batch_size'],
    num_workers = segmentation['num_workers']
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
    
    classification_transformation_obj = DataTransformation(classification_config, train_df, test_df, val_df, le)
    classification_train, classification_test, classification_val, class_map = classification_transformation_obj.initiate_transformation()

    # # model training
    # model_trainer_obj = ModelTrainer(model_trainer_config, classification_train, classification_val)
    # model_trainer_obj.train()

    # model evaluation
    model_eval_obj = ModelEvaluation(evlaution, classification_test, class_map)
    model_eval_obj.evaluate()

    # data transformation
    train_s = manifest_path / 'segmentation' / 'train.csv'
    test_s = manifest_path / 'segmentation' / 'test.csv'
    val_s = manifest_path / 'segmentation' / 'val.csv'

    train_df_s = pd.read_csv(train_s)
    test_df_s = pd.read_csv(test_s)
    val_df_s = pd.read_csv(val_s)

    segmentation_transformation_obj = Transformer(config = segmentation_config, train = train_df_s, test = test_df_s, val = val_df_s)
    segmentation_train, segmentation_test, segmentation_val = segmentation_transformation_obj.initiate_transformation()

    # # model training
    # model_trainer_obj = Trainer(model_trainer_config, segmentation_train, segmentation_val)
    # model_trainer_obj.train()

    # model evaluation
    model_eval_obj = Evaluator(evlaution, test_loader = segmentation_test)
