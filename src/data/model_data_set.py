from src.utils.rm_tree import rm_tree
from src.utils.partition_list import partition_list
from src.model.model_constants import *

import tarfile
import numpy as np
from random import shuffle
from pathlib import Path

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class ModelDataset:
    def __init__(self, model_info, config):
        self.config = config
        self.model_info = model_info

        self.image_feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/label/name': tf.io.FixedLenFeature([], tf.string),
            'image/label/value': tf.io.FixedLenFeature([], tf.int64),
            'image/data': tf.io.FixedLenFeature([], tf.string),
            'forces/data': tf.io.FixedLenFeature([], tf.string)
        }

    def _get_data_root_dirs(self):
        return {
            "raw": Path(self.config.get("data_raw_path", "data/raw")),
            "interim": Path(self.config.get("data_interim_path", "data/interim")),
            "processed": Path(self.config.get("data_processed_path", "data/processed"))
        }
    
    def _get_data_groups(self, model_data_files):
        groups = []

        data_files_mapped_name = {}
        for data_file in model_data_files:
            name = f"{data_file['dataSetLabel']}{data_file['dataLabel']}{data_file['dataSimType']}"
            data_files_mapped_name[name] = data_file            
        
        for data_set in self.model_info.model_data_sets:
            for label in DATA_LABELS:
                data_group = []
                for channel in self.model_info.model_channels:
                    data_group.append(data_files_mapped_name[f"{data_set}{label}{channel}"])
                groups.append(data_group)
        
        return groups

    def create_model_data_set(self):
        data_dirs = self._get_data_root_dirs()
        rm_tree(data_dirs["interim"])

        # Fetch config to run this script
        model_name = self.model_info.model_name

        image_height = self.config.get("image_height", 50)
        image_width = self.config.get("image_width", 50)

        data_train_split = self.config.get("data_train_split", 0.8)
        data_validation_split = self.config.get("data_validation_split", 0.8)

        # Load files
        model_data_files = self.model_info.get_model_data_files()
        
        data_groups = self._get_data_groups(model_data_files)
        num_groups = len(data_groups)

        
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        data_sets_writers = {
            "test": tf.io.TFRecordWriter(
                str(data_dirs["processed"]/f"{model_name}.test.tfrecords"),
                options=options
            ),
            "train": tf.io.TFRecordWriter(
                str(data_dirs["processed"]/f"{model_name}.train.tfrecords"),
                options=options
            ),
            "validation": tf.io.TFRecordWriter(
                str(data_dirs["processed"]/f"{model_name}.validation.tfrecords"),
                options=options
            )
        }
        
        for group_i, data_group in enumerate(data_groups):
            logger.info(f"Processing group {group_i}/{num_groups}")

            group_image_data = []
            group_label_data = []
            
            num_rows = int(data_group[0]["imageFileNumRows"])
            class_label = data_group[0]["dataLabel"]
            for data_file in data_group:
                data_file_path = data_dirs["raw"].joinpath(data_file["dataFile"])
                with tarfile.open(data_file_path) as archive:
                    image_file_name = archive.extractfile(data_file["imageFileName"])
                    label_file_name = archive.extractfile(data_file["labelFileName"])
                    
                    image_data = np.loadtxt(image_file_name, dtype=np.float32)
                    label_data = np.loadtxt(label_file_name, dtype=np.float32)

                    image_data = image_data.reshape(num_rows, image_height, image_width)

                    group_image_data.append(image_data)
                    group_label_data.append(label_data)
            
            image_channels_merged = np.stack(group_image_data, axis=-1)
            label_channels_merged = np.stack(group_label_data, axis=-1)

            shuffled_index = [i for i in range(num_rows)]
            shuffle(shuffled_index) # shuffled_index is now shuffled
            
            train, test = partition_list(shuffled_index, data_train_split)
            train, validate = partition_list(train, data_validation_split)

            data_sets = [
                ("train", train),
                ("test", test),
                ("validation", validate)
            ]

            for data_set_type, indexes in data_sets:
                writer = data_sets_writers[data_set_type]
                for i in indexes:
                    image = image_channels_merged[i]
                    label = label_channels_merged[i]
                    
                    features = {
                        "image/height": _int64_feature(image.shape[0]),
                        "image/width": _int64_feature(image.shape[1]),
                        "image/channels": _int64_feature(image.shape[2]),
                        "image/label/name": _bytes_feature(class_label.encode()),
                        "image/label/value": _int64_feature(DATA_LABELS[class_label]),
                        "image/data": _bytes_feature(image.tobytes()),
                        "forces/data": _bytes_feature(label.tobytes())
                    }

                    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tf_example.SerializeToString())

        for writer in data_sets_writers.values():
            writer.close()

    def load_model_data_set(self):
        data_dirs = self._get_data_root_dirs()
        model_name = self.model_info.model_name

        model_type = self.model_info.model_config['modelType']

        def _parse_data_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, self.image_feature_description)

        def _read_data_set(file_path):
            return tf.data.TFRecordDataset(file_path, compression_type="GZIP")
        
        def _parse_data(data_features):
            image_shape = tf.stack([
                data_features["image/height"],
                data_features["image/width"],
                data_features["image/channels"]
            ], 0)

            decoded_image = tf.io.decode_raw(data_features["image/data"], tf.float32)
            decoded_image = tf.reshape(decoded_image, image_shape)

            if model_type == "categorical":
                num_labels = len(DATA_LABELS)
                label = data_features["image/label/value"]
                one_hot_label = tf.one_hot(label, depth=num_labels)
                return decoded_image, one_hot_label
            elif model_type == "regression":
                raise NotImplementedError
            else:
                raise NotImplementedError

        def _prep_data_set(data_set):
            batch_size = self.config.get("data_batch_size", 32)

            data_set = data_set.map(_parse_data_function)
            data_set = data_set.map(_parse_data)
            data_set = data_set.repeat()
            data_set = data_set.shuffle(500)
            data_set = data_set.batch(batch_size)
            data_set = data_set.prefetch(buffer_size=AUTOTUNE)

            return data_set
        
        test =  _prep_data_set(_read_data_set(str(data_dirs["processed"]/f"{model_name}.test.tfrecords")))
        train = _prep_data_set(_read_data_set(str(data_dirs["processed"]/f"{model_name}.train.tfrecords")))
        valid = _prep_data_set(_read_data_set(str(data_dirs["processed"]/f"{model_name}.validation.tfrecords")))

        return test, train , valid

if __name__=="__main__":
    from src.config.load_workbook import load_workbook
    from src.model.model_info import ModelInfo

    config = {}

    model_config_dict, data_files_dict = load_workbook(config)

    model_info = ModelInfo("2_channel_categorical_gas_density_star_density_m1m2m3", model_config_dict, data_files_dict)
    model_dataset =  ModelDataset(model_info, config)

    model_dataset.create_model_data_set()
    test, _, _ = model_dataset.load_model_data_set()

    for image, label in test.take(1):
        print(image.shape)
        print(label.shape)

    # print(info.get_model_data_files())

