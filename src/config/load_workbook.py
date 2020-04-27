import pandas as pd

def load_workbook(config):
    workbook_path = config.get("model_config_path", "model_config.xlsx")

    models = pd.read_excel(workbook_path, sheet_name="Models")
    data_files = pd.read_excel(workbook_path, sheet_name="DataFiles")

    return models, data_files
