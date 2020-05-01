from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

from src.model.galaxy_model import GalaxyModelClassifier
from src.model.model_info import ModelInfo
from src.config.load_config import load_config
from src.config.load_workbook import load_workbook

import os
from pathlib import Path
from dotenv import load_dotenv

import tensorflow as tf

load_dotenv()


config = load_config("config.yml")
model_name = "2_channel_categorical_gas_density_star_density_m1m2m3"

mo = MongoObserver.create(url=os.environ.get("MONGO_CONNECTION_STRING"), db_name="db")
ex = Experiment(model_name)
ex.observers.append(mo)

ex.add_config(config)

@ex.automain
@LogFileWriter(ex)
def main(_run):
    
    config["epochs"] = 5
    model_config_dict, data_files_dict = load_workbook(config)

    model_info = ModelInfo(model_name, model_config_dict, data_files_dict)

    m = GalaxyModelClassifier(model_info, config)

    m.get_new_model()
    m.compile_model()
    
    t, tr, v = m.model_data_set.load_model_data_set()

    class CustomCallback(tf.keras.callbacks.BaseLogger):
        def __init__(self, run):
            self.run = run
            self.step = 0
            super(CustomCallback, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            print(f"On epoch {epoch}")
            for metric, val in logs.items():
                self.run.log_scalar(metric, val, epoch)
    
    history = m.train(tr, v, [CustomCallback(_run)])

    m.save_model()

    # import numpy as np
    # from PIL import Image

    # arr = np.random.randint(0,255,(100,100,3))
    # im = Image.fromarray(arr,'RGB')
    # out_path = "out.png"
    # im.save(out_path)
    # _run.add_artifact(out_path)