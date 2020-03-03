from pathlib import Path
import random
import logging

logger = logging.getLogger(__name__)

def partition_list(list_to_partition: list, percentage: float):
    split_index = round(len(list_to_partition)* percentage)
    return list_to_partition[:split_index], list_to_partition[split_index:]

def move_files_in_list(file_list, destination_dir):
    destination_dir.mkdir(exist_ok=True)
    for image_file in file_list:
        label = image_file.parent.stem
        label_dir = destination_dir / str(label)
        label_dir.mkdir(exist_ok=True)

        destination_file = label_dir/image_file.name
        logger.debug(f"Moving {image_file} to {destination_file}")
        image_file.replace(destination_file)

    logger.info(f"Moved {len(file_list)} images to {destination_dir}")
    

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
