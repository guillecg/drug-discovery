import os
import glob

from typing import List

import yaml


def load_yaml(path: str) -> dict:
    """Auxiliary function for loading YAMLs."""

    with open(path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


def get_test_data_paths(data_dir: str, data_type: str) -> List[str]:
    """Auxiliary function for retrieving test data paths."""

    return glob.glob(os.path.join(data_dir, f'*{data_type}'))
