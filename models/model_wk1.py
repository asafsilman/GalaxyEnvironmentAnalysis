import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.optimizers import Adadelta

from base.model import BaseModel

class ModelWeek1(BaseModel):
    def get_uncompiled_model(self):
        image_size = self.config.get("image_size", 50)
        num_classes = self.config.get("data_num_classes", 1)
        
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

    def compile_model(self, model):        
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adadelta(),
            metrics=['accuracy']
        )

        return model

    