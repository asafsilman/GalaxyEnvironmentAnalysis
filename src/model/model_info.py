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
        filter_function = lambda x: \
            self.model_config_dict[x]["modelName"]==self.model_name or \
            self.model_config_dict[x]["modelAltName"]==self.model_name

        _filter = filter(
            filter_function,
            self.model_config_dict
        )

        results = list(_filter)

        if len(results) != 1:
            raise Exception(f"Expected to find 1 result, found {len(results)} models with name {model_name}")

        return self.model_config_dict[results[0]]

    def _get_data_sets_info(self):
        data_sets = map(lambda x: (x, self.model_config[f"dataSet_{x}"]), AVAILABLE_DATASETS)
        
        return [d[0] for d in data_sets if d[1]]

    def _get_channels_info(self):
        channels = list(
            filter(lambda x: not pd.isna(x), 
                map(lambda x: self.model_config[f"channel{x}"], range(1, MAX_CHANNELS+1))
            )
        )

        return channels

    def get_model_data_files(self):
        data_sets = self.model_data_sets

        filter_function = lambda x: \
            self.data_files_dict[x]["dataSetLabel"] in data_sets and \
            self.data_files_dict[x]["dataSimType"] in self.model_channels

        return list(
            map(
                lambda x: self.data_files_dict[x],
                filter(filter_function, self.data_files_dict)
            )
        )

if __name__=="__main__":
    from src.config.load_workbook import load_workbook
    model_config_dict, data_files_dict = load_workbook({})

    info = ModelInfo("1_channel_categorical_gas_density_m1m2m3", model_config_dict, data_files_dict)

    print(info.get_model_data_files())
