import tensorflow as tf
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
import os

from pathlib import Path

AUTOTUNE = tf.data.experimental.AUTOTUNE
DEFAULT_IMAGE_SIZE = 50

def load_data_set(config, data_dir):
    image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
    batch_size = config.get("data_batch_size", 32)
    num_classes = config.get('data_num_classes', 1)

    class_names = np.array([item.name for item in data_dir.glob('*')])
    assert len(class_names) == num_classes, "Error len(class_names) != num_classes"

    # def gen():
    #     data_generator = ImageDataGenerator(
    #         rescale=1./255,
    #         zoom_range=0.2
    #     )

    #     train_generator = data_generator.flow_from_directory(
    #         data_dir,
    #         target_size=(image_size,image_size),
    #         color_mode='grayscale',
    #         shuffle=False
    #     )

    #     while True:
    #         for image, label in train_generator:
    #             for i in range(image.shape[0]):
    #                 yield image[i], label[i]

    
    # data_set = tf.data.Dataset.from_generator(
    #     gen,
    #     output_types=(tf.float32, tf.float32),
    #     output_shapes=(
    #         [image_size, image_size, 1],
    #         [num_classes]
    #     )
    # )

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=1)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [image_size, image_size])

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory

        return parts[-2] == class_names
    
    def process_path(file_path):
        label = get_label(file_path)
        
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    data_set = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    data_set = data_set.repeat()
    data_set = data_set.shuffle(500)
    data_set = data_set.batch(batch_size)
    data_set = data_set.prefetch(buffer_size=AUTOTUNE)
    return data_set