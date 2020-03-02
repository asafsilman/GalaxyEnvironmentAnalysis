"""
The data loader class helps load images and label data.
"""

import tensorflow as tf

class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def load_data_set(self, data_set_name):
        """Should return a tuple of train, test, and validation datasets

        If there is no validation set, it will be set to None
        If there is no test set, it will be set to None
        If there is no train set, it will be set to None
        
        Arguments:
            data_set_name {string} -- Data set name from config to load.

        Returns:
            tuple -- tuple of data sets like (train, test, validation).
        """
        raise NotImplementedError("Function not implemented")
