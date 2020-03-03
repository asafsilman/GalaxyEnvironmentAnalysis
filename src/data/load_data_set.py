import tensorflow as tf
import os

from pathlib import Path

AUTOTUNE = tf.data.experimental.AUTOTUNE
DEFAULT_IMAGE_SIZE = 50

def load_data_set(config, data_dir):
    image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
    batch_size = config.get("data_batch_size", 32)
    num_classes = len(config.get('data_labels', []))

    def gen():
        data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,)

        train_generator = data_generator.flow_from_directory(
            data_dir,
            target_size=(image_size,image_size),
            color_mode='grayscale'
        )

        for image, label in train_generator:
            for i in range(image.shape[0]):
                yield image[i],label[i]

    data_set = tf.data.Dataset.from_generator(
        gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            tf.TensorShape([image_size, image_size, 1]),
            tf.TensorShape([num_classes])
        )
    )

    data_set = data_set.batch(batch_size)
    data_set = data_set.prefetch(buffer_size=AUTOTUNE)
    return data_set