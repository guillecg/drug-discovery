import pytest

from modules.data.data_loaders import *
from modules.utils import load_yaml, get_test_data_paths


# IMPORTANT NOTE: all test files contain, by default, the 10 first records in
# 'data/Tox21/tox21_10k_data_all.sdf'
CONFIG = load_yaml(path='config.yaml')
TEST_DATA_DIR = CONFIG.get('TEST_DATA_DIR')


@pytest.fixture(scope='module')
def data_loader_manager():
    return DataLoaderManager()


@pytest.mark.slow
@pytest.mark.parametrize(
    'filters, expected_len', [
        ({}, 10),
        ({'SMILES': ''}, 0),
        ({'SMILES': 'CCN(CC)C(=S)SSC(=S)N(CC)CC'}, 1),
        pytest.param({'Non-existent column': ''}, 0, marks=pytest.mark.xfail)
    ]
)
def test_data_loader_manager(data_loader_manager, filters, expected_len):
    """This test fails if the filters do not work"""
    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        for path in paths:
            actual_len = len(
                data_loader_manager.load(path=path, filters=filters)
            )
            assert actual_len == expected_len


@pytest.mark.slow
def test_data_loaders_individually(data_loader_manager):
    """This test fails if there are not exactly 10 records"""
    for data_type, data_loader in data_loader_manager.data_loaders.items():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        for path in paths:
            assert len(data_loader().load(path=path)) == 10


@pytest.mark.slow
@pytest.mark.parametrize(
    'cls, kwargs', [
        (DataLoaderCSV, {'header': 0}),
        (DataLoaderExcel, {'header': 0}),
        (DataLoaderSDF, {'removeHs': False}),
        (DataLoaderSMILES, {'smilesColumn': 0}),
        pytest.param(DataLoaderCSV, {'NaN': True}, marks=pytest.mark.xfail),
        pytest.param(DataLoaderSDF, {'NaN': True}, marks=pytest.mark.xfail),
        pytest.param(DataLoaderSMILES, {'NaN': True}, marks=pytest.mark.xfail)
        # NOTE: pd.read_excel accepts kwargs, so it is worthless to test an
        # invalid argument, since it is always going to pass
        # pytest.param(DataLoaderExcel, {'NaN': True}, marks=pytest.mark.xfail)
    ]
)
def test_data_loader_kwargs(data_loader_manager, cls, kwargs):
    """This test fails if the kwargs are not properly defined"""
    for data_type, data_loader in data_loader_manager.data_loaders.items():
        if cls == data_loader:
            paths = get_test_data_paths(
                data_dir=TEST_DATA_DIR,
                data_type=data_type
            )

            for path in paths:
                assert len(data_loader().load(path=path, **kwargs)) == 10
