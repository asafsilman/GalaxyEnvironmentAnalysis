"""The purpose of this module is to load the configuration for a model from the
configuration excel workbook, in addition to the associated datafiles for the model.
"""

from src.model.model_constants import *
import pandas as pd

class ModelInfo:
    def __init__(self, model_name, model_config_dict, data_files_dict):
        self.model_name = model_name # The name of the model

        self.model_config_dict = model_config_dict # The whole model config dictionary
        self.data_files_dict = data_files_dict # The whole data file config dictionary

        self.model_config = self._get_model_config() # The config for the specific model
        self.model_data_sets = self._get_data_sets_info() # The datasets for the model
        self.model_channels = self._get_channels_info() # A list of channels the model will use

    def _get_model_config(self):
        # Try to load config from model_config_dict
        # Attept to match self.model_name to modelName or modelAltName
        filter_function = lambda x: \
            self.model_config_dict[x]["modelName"]==self.model_name or \
            self.model_config_dict[x]["modelAltName"]==self.model_name

        _filter = filter(
            filter_function,
            self.model_config_dict
        )

        results = list(_filter)

        # Expect there to be only one results from the above filter
        if len(results) != 1:
            raise Exception(f"Expected to find 1 result, found {len(results)} models with name {model_name}")

        # Return the config object
        return self.model_config_dict[results[0]]

    def _get_data_sets_info(self):
        """Get which datasets are specified in the model

        Returns:
            list -- A list of data set names in the model
        """

        # Return a iterator of (data_set_name, data_set_included)
        # data_set_included is a boolean
        data_sets = map(lambda x: (x, self.model_config[f"dataSet_{x}"]), AVAILABLE_DATASETS)
        
        return [data_set_name for data_set_name, data_set_included in data_sets if data_set_included]

    def _get_channels_info(self):
        """Get which channels are specified in the model

        Returns:
            list -- Channels which are used in model, in the correct order
        """

        channels = list(
            filter(lambda x: not pd.isna(x), 
                map(lambda x: self.model_config[f"channel{x}"], range(1, MAX_CHANNELS+1))
            )
        )

        return channels

    def get_model_data_files(self):
        """Get the data files config from self.data_files_dict

        Returns:
            list -- list of data set file configs which are used in the model
        """
        # Returns a list like ["m1", "m2", "m4" ...]
        data_sets = self.model_data_sets

        # Filter function which returns a boolean whether the dataset is included,
        # and the channel is included
        filter_function = lambda x: \
            self.data_files_dict[x]["dataSetLabel"] in data_sets and \
            self.data_files_dict[x]["dataSimType"] in self.model_channels

        return list( # Return as a list
            map(
                lambda x: self.data_files_dict[x], # Get the config dictionary
                filter(filter_function, self.data_files_dict) # Filter
            )
        )

if __name__=="__main__":
    from src.config.load_workbook import load_workbook
    model_config_dict, data_files_dict = load_workbook({})

    info = ModelInfo("1_channel_categorical_gas_density_m1m2m3", model_config_dict, data_files_dict)

    print(info.get_model_data_files())
