import pytest

from collections import defaultdict

from typing import List

import pandas as pd

from modules.data.data_loaders import DataLoaderManager
from modules.utils import load_yaml, get_test_data_paths


@pytest.fixture(scope='session')
def data_loader_manager() -> DataLoaderManager:
    return DataLoaderManager()


@pytest.fixture(scope='session')
def config_path() -> str:
    return 'config.yml'


@pytest.fixture(scope='session')
def data_paths_dict(
        data_loader_manager: DataLoaderManager,
        config_path: str
) -> dict:
    """Fixture containing the directory path in which test data files are."""

    # IMPORTANT NOTE: all test files contain, by default,
    # the 10 first records in 'data/Tox21/tox21_10k_data_all.sdf'

    # Get test data directory from test config
    config = load_yaml(path=config_path)
    test_data_dir = config.get('TEST_DATA_DIR')

    paths_dict = defaultdict(list)

    # For each supported format in DataLoaderManager
    for data_format in data_loader_manager.data_loaders.keys():
        # Discover all files for that supported format
        paths = get_test_data_paths(
            data_dir=test_data_dir,
            data_type=data_format
        )

        paths_dict[data_format].extend(paths)

    return paths_dict


@pytest.fixture(scope='session')
def data_list(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict
) -> List[pd.DataFrame]:
    """Fixture containing all test data for tests disposal."""

    data_list = []

    # For each file in the test data folder
    for path in data_paths_dict.values():
        # Load data and append to list
        data = data_loader_manager.load(path=path)
        data_list.append(data)

    return data_list
