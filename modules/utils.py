import os
import glob

import yaml


def load_yaml(path: str):
    """Auxiliary function for loading YAMLs."""

    with open(path) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        return yaml.load(file, Loader=yaml.FullLoader)


def get_test_data_paths(data_dir: str, data_type: str) -> list:
    """Auxiliary function for getting test data paths."""
    return glob.glob(os.path.join(data_dir, f'*{data_type}'))
