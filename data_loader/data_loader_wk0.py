"""Dataloader sample for week 0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from base.data_loader import BaseDataLoader

DEFAULT_IMAGE_SIZE = 50

class DataLoaderWeek0(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)


    def _load_data_using_test_loader(self, data_set):
        image_data_file = data_set["image_data"]
        labal_data_file = data_set["label_data"]
        
        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)

        image_data = pd.read_csv(image_data_file,header=None, dtype=np.float64, squeeze=True)
        image_data = image_data.to_numpy()
        num_images = image_data.shape[0]//(image_size*image_size)
        image_data = image_data.reshape(num_images, image_size, image_size, 1)

        label_data = pd.read_csv(labal_data_file, header=None, skiprows=1, sep="\s+")
        label_data = label_data.to_numpy()

        return image_data, label_data

    def _load_data_using_default_loader(self, data_set):
        image_data_file = data_set["image_data"]
        ram_label_data_file = data_set["label_data_RAM"]
        tid_label_data_file = data_set["label_data_TID"]

        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)

        image_data = pd.read_csv(image_data_file,header=None, dtype=np.float64, squeeze=True)
        image_data = image_data.to_numpy()
        num_images = image_data.shape[0]//(image_size*image_size)
        image_data = image_data.reshape(num_images, image_size, image_size, 1)

        ram_data = pd.read_csv(ram_label_data_file, header=None, dtype=np.float64, squeeze=True)
        tid_data = pd.read_csv(tid_label_data_file, header=None, dtype=np.float64, squeeze=True)

        label_data = pd.DataFrame(data={"RAM_Data": ram_data, "TID_Data": tid_data})

        return image_data, label_data

    def _get_data_augmentor(self, image_data):
        data_augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     zoom_range=0.2
        ) # create image data augmentor
        data_augmentor.fit(image_data)

        return data_augmentor

    def _get_data_generator(self, X, Y, use_augmentation):
        augmentor = self._get_data_augmentor(X)

        def _generator():
            if use_augmentation:
                for x, y in augmentor.flow(X, Y, shuffle=False):
                    for i in range(x.shape[0]):
                        yield x[i],y[i]
            else:
                for i in range(X.shape[0]):
                    yield X[i],Y[i]
        return _generator
    
    def _create_data_set(self, X, y, use_augmentation=True, repeat=True):
        data_generator = self._get_data_generator(X, y, use_augmentation)
        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)

        data_set = tf.data.Dataset.from_generator(
            data_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                tf.TensorShape([image_size, image_size, 1]),
                tf.TensorShape([2])
            )
        )

        
        if repeat:
            data_set = data_set.repeat()
        data_set = data_set.batch(self.config.get("data_batch_size", 32))
        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        data_set = data_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        return data_set

    def load_data_set(self, data_set_name):
        data_set = self.config["data_sets"].get(data_set_name)
        if data_set is None:
            raise ValueError(f"Could not find data set ({data_set_name}).")

        loader = data_set.get("loader", "default")

        if loader == "test_data":
            image_data, label_data = self._load_data_using_test_loader(data_set)

            test_data_set = self._create_data_set(
                image_data,
                label_data,
                use_augmentation=False,
                repeat=False
            )
            return None, test_data_set, None
        elif loader == "default":
            data_train_split = self.config.get("data_train_split", 0.8)
            image_data, label_data = self._load_data_using_default_loader(data_set)

            X_train, X_test, y_train, y_test = train_test_split(image_data, label_data, train_size=data_train_split)

            train_data_set = self._create_data_set(X_train, y_train)
            validation_data_set = self._create_data_set(X_test, y_test)
            return train_data_set, None, validation_data_set
            
        else:
            raise ValueError(f"loader value set to invalid value in config ({loader}).")

if __name__=="__main__":
    from utils.load_config import load_config
    config = load_config("config/default.yml")

    dl = DataLoaderWeek0(config)

    for image, label in dl.get_test_data_set().take(5):
        print(label.shape)

