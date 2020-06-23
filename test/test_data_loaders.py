import os
import glob

import pytest

from modules.data.data_loaders import *


# IMPORTANT NOTE: all test files contain the 10 first records in
# 'data/Tox21/tox21_10k_data_all.sdf'
TEST_DATA_DIR = 'data/'
DATA_TYPES = {
    'csv': DataLoaderCSV,
    'xls': DataLoaderExcel,
    'xlsx': DataLoaderExcel,
    'smi': DataLoaderSMILES,
    'sdf': DataLoaderSDF
}


def get_test_data_paths(data_dir: str, data_type: str) -> list:
    return glob.glob(os.path.join(data_dir, f'*.{data_type}'))


@pytest.mark.slow
@pytest.mark.parametrize('data_type', DATA_TYPES.keys())
def test_get_test_data_paths(data_type: str):
    """This test fails if no test file with the required format is found."""
    assert get_test_data_paths(data_dir=TEST_DATA_DIR, data_type=data_type)


@pytest.mark.slow
@pytest.mark.parametrize('data_type', DATA_TYPES.keys())
def test_data_loaders_bulk(data_type: str):
    """This test fails if there are not exactly 10 records"""
    data_loader = DATA_TYPES[data_type]()
    paths = get_test_data_paths(data_dir=TEST_DATA_DIR, data_type=data_type)

    assert [len(data_loader.load(path)) == 10 for path in paths]
