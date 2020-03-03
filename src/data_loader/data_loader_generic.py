import tarfile
from pathlib import Path

from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from base.data_loader import BaseDataLoader

DEFAULT_IMAGE_SIZE = 50

class DataLoaderGeneric(BaseDataLoader):
    def _load_data(self, data_set):
        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)

        data_files = data_set.get("data_files")
        
        label_data_sets = []
        image_data_sets = []

        for data_file_info in data_files:
            data_file_path = data_file_info.get("file")
            image_file_name = data_file_info.get("image_file_name", "m1.dir/2dft.dat")
            label_file_names = data_file_info.get("label_file_names", "m1.dir/2dftn1.dat")
            if isinstance(label_file_names, str):
                label_file_names = [label_file_names]


            if Path(data_file_path).exists():
                with tarfile.open(data_file_path) as tar:
                    # Extract data files (image and labels)
                    label_files = []

                    for label_file_name in label_file_names:
                        label_files.append(tar.extractfile(label_file_name))
                    image_file = tar.extractfile(image_file_name)

                    # Load data using np.loadtxt
                    label_data_list = []
                    for label_file in label_files:
                        label_data = np.loadtxt(label_file, dtype=np.float32)
                        n_rows = label_data.shape[0]
                        label_data = label_data.reshape(n_rows, 1)
                        label_data_list.append(label_data)

                    image_data = np.loadtxt(image_file, dtype=np.float32)

                    # Count number of rows
                    n_rows = label_data.shape[0]

                    # Reshape data using numpy reshape
                    label_data = label_data.reshape(n_rows, 1)
                    image_data = image_data.reshape(n_rows, image_size, image_size, 1)

                    # Adjust label data according to rules in data config
                    labels = data_file_info.get("labels", {})
                    default_label = data_file_info.get("default_label", 0)
                    
                    for i in range(n_rows):
                        if label_data[i][0] in labels: # If label is in list of labels
                            label_data[i][0] = labels[label_data[i][0]] # Assign label to value in config
                        else:
                            label_data[i] = default_label # Else assign default value

                    # Add data to datasets, these values will all be concatonated in the final stage
                    label_data_sets.append(label_data)
                    image_data_sets.append(image_data)
            else:
                raise FileNotFoundError(f"Cannot find file: {data_file_path}")

        labal_data_set_merged = np.concatenate(label_data_sets).astype(np.int)
        image_data_set_merged = np.concatenate(image_data_sets)

        return image_data_set_merged, labal_data_set_merged

    def _create_data_set(self, X, y, repeat=True):
        def _gen():
            for i in range(X.shape[0]):
                yield X[i],y[i]
    
        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)
        
        data_set = tf.data.Dataset.from_generator(
            _gen,
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape([image_size, image_size, 1]),
                tf.TensorShape([1])
            )
        )

        if repeat: data_set = data_set.repeat()
        data_set = data_set.shuffle(500)
        data_set = data_set.batch(self.config.get("data_batch_size", 32))
        data_set = data_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return data_set
    
    def load_data_set(self, data_set_name):
        data_set = self.config["data_sets"].get(data_set_name)
        if data_set is None:
            raise ValueError(f"Could not find data set ({data_set_name}).")

        data_train_split = self.config.get("data_train_split", 0.8)
        data_validation_split = self.config.get("data_validation_split", 0.8)

        image_data, label_data =  self._load_data(data_set)
        
        X_train, X_test, y_train, y_test = train_test_split(image_data, label_data, train_size=data_train_split)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, train_size=data_validation_split)

        train_data_set = self._create_data_set(X_train, y_train)
        test_data_set = self._create_data_set(X_test, y_test, repeat=False)
        validation_data_set = self._create_data_set(X_validate, y_validate)

        return train_data_set, test_data_set, validation_data_set

if __name__=="__main__":
    config = {
        "data_sets": {
            "test": {
                "data_files": [
                    {
                        "file": "data/A.i.m1.dir.tar.gz",
                        "labels": {},
                        "default_label": 2
                    },
                    {
                        "file": "data/A.m1.dir.tar.gz",
                        "labels": {0: 0},
                        "default_label": 1
                    }
                ]
            }
        }
    }

    train_data_set, test_data_set, validation_data_set = DataLoaderGeneric(config).load_data_set("test")

    from utils.show_image import show_image
    show_image(train_data_set)
