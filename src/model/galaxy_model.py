import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adadelta

from src.data.model_data_set import ModelDataset
from src.model.model_constants import DATA_LABELS
from src.model.get_ROC_curve import get_ROC_curve

logger = logging.getLogger(__name__)

class GalaxyModel:
    def __init__(self, model_info, config):
        self.model_info = model_info
        self.config = config
        self.model_data_set = ModelDataset(model_info, config)
        self.model = None

    def get_new_model(self):
        raise NotImplementedError

    def compile_model(self):
        raise NotImplementedError

    def load_model(self):
        model_directory = self.config["model_path"]
        model_name = self.model_info.model_name

        return  tf.keras.models.load_model(model_name)

    def save_model(self):
        if self.model is None:
            raise ValueError("No model is defined")

        model_directory = self.config["model_path"]
        model_name = self.model_info.model_name
        
        save_path = Path(model_directory) / model_name

        self.model.save(str(save_path))

    def _get_training_callbacks(self, callbacks):
        _callbacks = []

        if callbacks is not None:
            for callback in callbacks:
                _callbacks.append(callback)

        model_name = self.model_info.model_name
        enable_tensorboard = self.config.get('enable_tensorboard', False)
        enable_early_stopping = self.config.get('enable_earlystopping', False)

        if enable_tensorboard:
            # Setup tensorboard logs
            log_dir = Path(self.config.get('log_dir', 'logs/fit'))
            
            today_date_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir=log_dir / f"{model_name}-{today_date_time_str}"

            _callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
            )
        if enable_early_stopping:
            _callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            )
        
        return _callbacks

    def train(self, train_data_set, validation_data_set, callbacks=None):
        epochs = self.config["epochs"]
        callbacks = self._get_training_callbacks(callbacks)
        
        return self.model.fit(
            train_data_set,
            steps_per_epoch=50,
            epochs=epochs,
            validation_data=validation_data_set,
            validation_steps=50,
            verbose=0,
            callbacks=callbacks
        )

class GalaxyModelClassifier(GalaxyModel):
    def get_new_model(self):
        # Get model parameters
        image_height = self.config["image_height"]
        image_width = self.config["image_width"]
        image_channels = self.model_info.model_config.get("numChannels")
        
        # Build model architecture
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(image_height, image_width, image_channels)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(DATA_LABELS), activation='softmax'))

        # return model
        self.model = model

    def compile_model(self):
        if self.model is None:
            raise ValueError("No model is defined")

        self.model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=Adadelta(),
            metrics=['MSE', 'accuracy', 'AUC']
        )

    def _get_predictions(self, testing_data_set):
        predict, correct = [], []
        predict_scores, correct_scores = [], []
        
        for image, label in testing_data_set:
            predictions = self.model.predict(image)

            for i in range(predictions.shape[0]):
                predict.append(np.argmax(predictions[i]))
                correct.append(np.argmax(label[i]))

                predict_scores.append(predictions[i])
                correct_scores.append(label[i])

        return (predict, predict_scores, correct, correct_scores)

    def evaluate_model(self, testing_data_set, ex):
        if self.model is None:
            raise ValueError("No model is defined")

        predict, predict_scores, correct, correct_scores = self._get_predictions(testing_data_set)

        ROC_curve = get_ROC_curve(predict_scores, correct_scores)

        ex.add_artifact(ROC_curve.name, name="ROC_Curve.png")
        ROC_curve.close()

if __name__=="__main__":
    from src.config.load_workbook import load_workbook
    from src.config.load_config import load_config
    from src.model.model_info import ModelInfo
    
    config = load_config("config.yml")
    config["epochs"] = 10
    model_config_dict, data_files_dict = load_workbook(config)

    model_info = ModelInfo("2_channel_categorical_gas_density_star_density_m1m2m3", model_config_dict, data_files_dict)

    m = GalaxyModelClassifier(model_info, config)

    m.get_new_model()
    m.compile_model()
    
    t, tr, v = m.model_data_set.load_model_data_set()

    m.train(tr, v)
