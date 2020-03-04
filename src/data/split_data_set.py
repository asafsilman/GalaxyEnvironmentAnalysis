from pathlib import Path
import random
import logging

from src.utils.move_files_in_list import move_files_in_list
from src.utils.partition_list import partition_list

logger = logging.getLogger(__name__)    

# Assume that the file names that are hashed are sufficiently random
def split_data_set(config):
    data_interim_path = Path(config.get('data_interim_path', 'data/interim'))
    data_processed_path = Path(config.get('data_processed_path', 'data/processed'))

    data_train_split = config.get('data_train_split', 0.8)
    data_validation_split = config.get('data_validation_split', 0.8)

    image_files = list(data_interim_path.glob("**/*.png"))
    random.shuffle(image_files)

    train, test = partition_list(image_files, data_train_split)
    train, validate = partition_list(train, data_validation_split)

    move_files_in_list(test, data_processed_path/"test")
    move_files_in_list(train, data_processed_path/"train")
    move_files_in_list(validate, data_processed_path/"validate")
