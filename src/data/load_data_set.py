# DEPRICATED

import tensorflow as tf
import numpy as np

import os
import logging

from pathlib import Path

AUTOTUNE = tf.data.experimental.AUTOTUNE
DEFAULT_IMAGE_SIZE = 50

logger = logging.getLogger(__name__)

def load_data_set(config, data_dir):
    image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
    batch_size = config.get("data_batch_size", 32)
    num_classes = config.get('data_num_classes', 1)

    model_type = config.get("model_type", "1_channel")
    channels = 1 # initialisation

    if model_type=="2_channel":
        channels=3
    else:
        channels=1

    class_names = np.array(
        sorted([item.name for item in data_dir.glob('*')]) # Path.glob can have different orderings on different systems
    )
    assert len(class_names) == num_classes, f"Error len(class_names) != {num_classes}"
    logger.debug(f"class names are: {class_names}")

    def decode_img(img, channels=1):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [image_size, image_size])

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory

        return tf.cast(parts[-2] == class_names, tf.float32)
    
    def process_path(file_path):
        label = get_label(file_path)
        
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img, channels)
        return img, label

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    data_set = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    data_set = data_set.repeat()
    data_set = data_set.shuffle(500)
    data_set = data_set.batch(batch_size)
    data_set = data_set.prefetch(buffer_size=AUTOTUNE)
    return data_set