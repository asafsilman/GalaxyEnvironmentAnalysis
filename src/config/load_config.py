import json
import pathlib
import yaml

def load_config(config_path):
    path = pathlib.Path(config_path)

    if path.is_file() and path.exists():
        if path.suffix == ".json":
            with open(path) as f:
                return json.load(f)
        elif path.suffix in (".yaml", ".yml"):
            with open(path) as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        else:
            raise ValueError(f"Config file extension not accepted.")
    else:
        raise FileNotFoundError(f"Config file not found. Path={config_path}")
