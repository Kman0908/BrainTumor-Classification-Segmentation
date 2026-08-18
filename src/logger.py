import os
import logging 
from datetime import datetime

LOG_FILE = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'

logs = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs, exist_ok = True)

LOG = os.path.join(logs, LOG_FILE)

logging.basicConfig(
    level = logging.INFO,
    filename = LOG,
    format = '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s'
)

logging.getLogger(__name__)