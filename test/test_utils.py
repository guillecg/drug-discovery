import pytest

from modules.data.data_loaders import DataLoaderManager
from modules.utils import load_yaml, get_test_data_paths


# IMPORTANT NOTE: all test files contain, by default, the 10 first records in
# 'data/Tox21/tox21_10k_data_all.sdf'
CONFIG = load_yaml(path='config.yaml')
TEST_DATA_DIR = CONFIG.get('TEST_DATA_DIR')


@pytest.fixture(scope='module')
def data_loader_manager():
    return DataLoaderManager()


@pytest.mark.parametrize('path', ['config.yaml'])
def test_load_yaml(path):
    """This test fails if the result returned is not a dict."""
    yaml_file = load_yaml(path=path)

    assert isinstance(yaml_file, dict)


def test_get_test_data_paths(data_loader_manager):
    """This test fails if no test file with the required format is found."""
    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        assert paths
        assert isinstance(paths, list)
