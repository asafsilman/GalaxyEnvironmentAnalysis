import logging
import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.optimizers import Adadelta

from tensorflow.python.eager import profiler

from src.data.load_data_set import load_data_set

DEFAULT_IMAGE_SIZE = 50

logger = logging.getLogger(__name__)

def get_new_model(image_size, num_classes, channels=1):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(image_size, image_size, channels)))
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

def train_model(config, new_model: bool, save_model_flag: bool):
    image_size = config.get('image_size', DEFAULT_IMAGE_SIZE)
    num_classes = config.get('data_num_classes', 1)

    model_directory = Path(config.get('model_directory', 'models'))
    model_name = config.get('model_name', 'unnamed_model')

    data_processed_path = Path(config.get('data_processed_path', 'data/processed')) / model_name

    if new_model:
        logger.info("Creating new model")
        model_type = config.get("model_type", "1_channel")
        if model_type == "1_channel":
            model = get_new_model(image_size, num_classes, channels=1)
        elif model_type == "2_channel":
            model = get_new_model(image_size, num_classes, channels=3) # tensorflow only accepts 1, 3 or 4 channel images
    else:
        logger.info(f"Loading model {model_name} from {model_directory}")
        model = load_model(model_name, model_directory)

    train_data_set = load_data_set(config, data_processed_path/'train')
    validation_data_set = load_data_set(config, data_processed_path/'validate')

    epochs = config.get("epochs", 100)

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=Adadelta(),
        metrics=['MSE', 'accuracy', 'AUC']
    )
    
    callbacks = []

    enable_tensorboard = config.get('enable_tensorboard', False)
    if enable_tensorboard:
        # Setup tensorboard logs
        log_dir = Path(config.get('log_dir', 'logs/fit'))
        
        today_date_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir=log_dir / f"{model_name}-{today_date_time_str}"
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
        )

        profiler.start()
    if config.get('enable_earlystopping', False):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        )

    

    if save_model_flag:
        logger.info(f"Saving model as {model_name}")
        save_model(model, model_name, model_directory)
    else:
        logger.info("Discarding training. save_model_flag set to False")

    if enable_tensorboard:
        profiler_result = profiler.stop()
        profiler.save(str(log_dir), profiler_result)
