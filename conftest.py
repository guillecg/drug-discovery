import pytest

from collections import defaultdict

from typing import List

import pandas as pd

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split

from modules.data_loaders import DataLoaderManager
from modules.utils import load_yaml, get_test_data_paths


@pytest.fixture(scope='session')
def seed():
    return 42


@pytest.fixture(scope='session')
def data_loader_manager() -> DataLoaderManager:
    return DataLoaderManager()


@pytest.fixture(scope='session')
def config() -> dict:
    return load_yaml(path='tests/config.yml')


@pytest.fixture(scope='session')
def data_paths_dict(
        data_loader_manager: DataLoaderManager,
        config: dict
) -> dict:
    """Fixture containing all the test data file paths for their corresponding
    formats.
    """

    # IMPORTANT NOTE: all test files contain, by default,
    # the 10 first records in 'data/Tox21/tox21_10k_data_all.sdf'

    # Get test data directory from test config
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

    # For each supported format in DataLoaderManager
    for paths in data_paths_dict.values():
        # For each file with the supported format in the test data folder
        for path in paths:
            # Load data and append to list
            data = data_loader_manager.load(path=path)
            data_list.append(data)

    return data_list


@pytest.fixture(scope='session')
def data_classification(seed: int, test_size: float = 0.33):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='session')
def data_regression(seed: int, test_size: float = 0.33):
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test
