import yaml
from pathlib import Path
from src.logger import logging
from src.exception import CustomException

def read_yaml(path: Path) -> dict:
    try:
        with open(path, 'r') as f:
            content = yaml.safe_load(f)
            logging.info('.yaml File loaded {path}')
            return content
    except Exception as e:
        raise CustomException(e)