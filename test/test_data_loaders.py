import os
import glob

import pytest

from modules.data.data_loaders import *


# IMPORTANT NOTE: all test files contain, by default, the 10 first records in
# 'data/Tox21/tox21_10k_data_all.sdf'
TEST_DATA_DIR = 'data/'


@pytest.fixture(scope='module')
def data_loader_manager():
    return DataLoaderManager()


def get_test_data_paths(data_dir: str, data_type: str) -> list:
    return glob.glob(os.path.join(data_dir, f'*{data_type}'))


@pytest.mark.slow
def test_get_test_data_paths(data_loader_manager):
    """This test fails if no test file with the required format is found."""
    for data_type in data_loader_manager.data_loaders.keys():
        assert get_test_data_paths(data_dir=TEST_DATA_DIR, data_type=data_type)


@pytest.mark.slow
def test_data_loaders_individually(data_loader_manager):
    """This test fails if there are not exactly 10 records"""
    for data_type, data_loader in data_loader_manager.data_loaders.items():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        for path in paths:
            assert len(data_loader_manager.load(path)) == 10


@pytest.mark.slow
def test_data_loader_manager(data_loader_manager):
    """This test fails if there are not exactly 10 records"""
    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        for path in paths:
            assert len(data_loader_manager.load(path)) == 10
