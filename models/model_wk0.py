from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import CosineSimilarity
from tensorflow.keras.optimizers import Adadelta

from base.model import BaseModel

class ModelWeek0(BaseModel):
    def get_uncompiled_model(self):
        image_size = self.config.get("image_size", 50)
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(image_size, image_size, 1)))
                        # input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='linear'))

        return model

    def compile_model(self, model):
        cosine_similarity = CosineSimilarity(axis=1)
        
        model.compile(
            loss='mean_squared_error',
            optimizer=Adadelta(),
            metrics=['MSE', 'MAE', cosine_similarity]
        )

        return model

    