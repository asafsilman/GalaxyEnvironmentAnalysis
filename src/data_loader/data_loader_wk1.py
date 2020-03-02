import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from base.data_loader import BaseDataLoader

DEFAULT_IMAGE_SIZE = 50

class DataLoaderWeek1(BaseDataLoader):
    def _load_data(self, data_set):
        image_data_files = data_set["image_data"]
        label_data_file = data_set["label_data"]

        image_size = self.config.get("image_size", DEFAULT_IMAGE_SIZE)
        num_classes = self.config.get("data_num_classes")

        image_data_array = []
        for image_data_file in image_data_files:
            image_data = pd.read_csv(image_data_file,header=None, dtype=np.float64, squeeze=True)
            image_data = image_data.to_numpy()
            image_data_array.append(image_data)

        if len(image_data_array) == 1:
            image_data = image_data_array[0]
        else:
            image_data = image_data_array[0]
            for image_data_new in image_data_array[1:]:
                image_data = np.concatenate((image_data, image_data_new))

        num_images = image_data.shape[0]//(image_size*image_size)
        image_data = image_data.reshape(num_images, image_size, image_size, 1)

        label_data = pd.read_csv(label_data_file, header=None, dtype=np.int, squeeze=True)
        label_data = label_data.to_numpy() - 1 # make labels start from 0
        
        label_data = label_data.reshape(label_data.shape[0], 1) # Must reshape to work with tensorflow dataset
        
        labels_data = tf.keras.utils.to_categorical(label_data, num_classes)

        return image_data, label_data

    def create_data_set(self, X, y, repeat=True):
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
        image_data, label_data = self._load_data(data_set)

        X_train, X_test, y_train, y_test = train_test_split(image_data, label_data, train_size=data_train_split)
        X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, train_size=data_validation_split)

        train_data_set = self.create_data_set(X_train, y_train)
        test_data_set = self.create_data_set(X_test, y_test, repeat=False)
        validation_data_set = self.create_data_set(X_validate, y_validate)

        return train_data_set, test_data_set, validation_data_set

if __name__=="__main__":
    config = {
        "data_sets":
            {
                "test": {
                    "image_data": ["data/Week1/2dft.dat", "data/Week1/2dft.i.dat"],
                    "label_data": "data/Week1/2dftn.dat"
                }
            },
        "image_size": 50,
        "data_num_classes": 3
    }

    data_loader = DataLoaderWeek1(config)
    train_data_set, test_data_set, validation_data_set = data_loader.load_data_set("test")
    ii = 0
    for i, l in train_data_set.take(50):
        print(ii)
        ii+=1
    print(test_data_set.take(50))