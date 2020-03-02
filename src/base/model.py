"""
Base model class which may be helpful to implement for future models
"""

from tensorflow.keras.models import model_from_json

class BaseModel(object):
    def __init__(self, config):
        self.config = config

    def get_uncompiled_model(self):
        raise NotImplementedError("Function not implemented")

    def compile_model(self, model):
        raise NotImplementedError("Function not implemented")

    def save_model_and_weights_to_disk(self, model, model_file, weights_file):
        model_json = model.to_json()
        with open(model_file, "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(weights_file)

    def load_model_and_weights_from_disk(self, model_file, weights_file):
        with open(model_file) as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights(weights_file)

        return loaded_model
