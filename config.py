from pathlib import Path
from ruamel.yaml import YAML
from typing import Dict
import pandas as pd


def read_config(cfg_path: Path) -> Dict:
    if cfg_path.exists():
        with cfg_path.open("r") as fp:
            yaml = YAML(typ="safe")
            cfg = yaml.load(fp)
    else:
        raise FileNotFoundError(cfg_path)

    cfg = parse_config(cfg)

    return cfg


def parse_config(cfg: Dict) -> Dict:
    """Parse a config Dict object (keys: values)

    Args:
        cfg (Dict): [description]

    Returns:
        Dict: Parsed config dictionary
    """
    for key, val in cfg.items():
        # convert all path strings to PosixPath objects
        if any([x in key for x in ["dir", "path", "file"]]):
            if (val is not None) and (val != "None"):
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # convert Dates to pandas Datetime indexs
        elif key.endswith("_date"):
            if isinstance(val, list):
                temp_list = []
                for elem in val:
                    temp_list.append(pd.to_datetime(elem, format="%d/%m/%Y"))
                cfg[key] = temp_list
            else:
                cfg[key] = pd.to_datetime(val, format="%d/%m/%Y")

        # None to be interpreted as empty lists
        elif any(
            key == x
            for x in ["static_inputs", "camels_attributes", "hydroatlas_attributes"]
        ):
            if val is None:
                cfg[key] = []
        else:
            pass

    # Add more config parsing if necessary

    return cfg
