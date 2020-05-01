import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver

import os

class CustomCallback(tf.keras.callbacks.BaseLogger):
    def __init__(self, run):
        self.run = run
        self.step = 0
        super(CustomCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(f"On epoch {epoch}")
        for metric, val in logs.items():
            self.run.log_scalar(metric, val, epoch)


def run_model_experiment(model, config):
    mo = MongoObserver.create(url=os.environ.get("MONGO_CONNECTION_STRING"), db_name="db")

    ex = Experiment(model.model_info.model_name)
    ex.observers.append(mo)

    ex.add_config(config)



    @ex.automain
    def main(_run):
        model.get_new_model()
        model.compile_model()
        
        test, train, validate = model.model_data_set.load_model_data_set()
 
        history = model.train(train, validate, [CustomCallback(_run)])

        model.save_model()
        model.evaluate_model(test, ex)

    ex.run()