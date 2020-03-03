from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.optimizers import Adadelta

from src.data.load_data_set import load_data_set

DEFAULT_IMAGE_SIZE = 50

def get_new_model(image_size, num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(image_size, image_size, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def load_model(model_name, model_directory):
    model_file = model_directory/f"{model_name}.json"
    model_weights = model_directory/f"{model_name}.h5"
    
    with open(model_file) as json_file:
        loaded_model = model_from_json(json_file.read())
    
    # load weights into new model
    loaded_model.load_weights(str(model_weights))

    return loaded_model

def save_model(model, model_name, model_directory):
    model_file = model_directory/f"{model_name}.json"
    model_weights = model_directory/f"{model_name}.h5"
    
    with open(model_file, 'w') as json_file:
        json_file.write(model.to_json())
    
    model.save_weights(str(model_weights))

def train_model(config, new_model: bool):
    image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
    num_classes = config.get('data_num_classes', 1)
    data_processed_path = Path(config.get('data_processed_path', 'data/processed'))

    model_directory = Path(config.get('model_directory', 'models'))
    model_name = config.get('model_name', 'unnamed_model')

    if new_model:
        model = get_new_model(image_size, num_classes)
    else:
        model = load_model(model_name, model_directory)

    train_data_set = load_data_set(config, data_processed_path/'train')
    validation_data_set = load_data_set(config, data_processed_path/'validate')

    epochs = min(config.get("epochs", 100), 10)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=Adadelta(),
        metrics=['MSE']
    )
    
    model.fit(
        train_data_set,
        steps_per_epoch=50,
        epochs=epochs,
        validation_data=validation_data_set,
        validation_steps=50,
        verbose=1,
    )

    save_model(model, model_name, model_directory)
