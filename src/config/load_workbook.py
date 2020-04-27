import pandas as pd

def data_files_df_to_dict(data_files_df):
    return data_files_df.to_dict(orient="index")

def load_workbook(config):
    workbook_path = config.get("model_config_path", "model_config.xlsx")

    models = pd.read_excel(workbook_path, sheet_name="Models")
    data_files = pd.read_excel(workbook_path, sheet_name="DataFiles")

    models_dict = data_files_df_to_dict(models)
    data_dict = data_files_df_to_dict(data_files)

    return models_dict, data_dict
