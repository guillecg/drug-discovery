import pytest

from modules.base import BaseDataLoader

from modules.data_loaders import (
    DataLoaderManager,
    DataLoaderCSV,
    DataLoaderExcel,
    DataLoaderSMILES,
    DataLoaderSDF
)


@pytest.mark.slow
@pytest.mark.parametrize(
    'filters, expected_len', [
        ({}, 10),
        ({'SMILES': ''}, 0),
        ({'SMILES': 'CCN(CC)C(=S)SSC(=S)N(CC)CC'}, 1),
        ({'SMILES': [
            'CCN(CC)C(=S)SSC(=S)N(CC)CC',
            'CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1'
        ]}, 2),
        ({'SMILES': ['', 39]}, 0),
        pytest.param({'Non-existent column': ''}, 0, marks=pytest.mark.xfail)
    ]
)
def test_data_loader_manager_filters(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict,
        filters: dict,
        expected_len: int
) -> None:
    """This test fails if the filters in DataLoaderManager.load do not work."""

    # For each supported data format in DataLoaderManager
    for data_format, data_loader in data_loader_manager.data_loaders.items():
        # Retrieve paths from fixture
        paths = data_paths_dict[data_format]

        # For each file for the supported data format
        for path in paths:
            actual_len = len(
                data_loader_manager.load(path=path, filters=filters)
            )
            assert actual_len == expected_len


@pytest.mark.slow
def test_data_loaders_individually(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict
) -> None:
    """This test fails if there are not exactly 10 records for each data
    format using DataLoaderManager.
    """

    # For each supported data format in DataLoaderManager
    for data_format, data_loader in data_loader_manager.data_loaders.items():
        # Retrieve paths from fixture
        paths = data_paths_dict[data_format]

        # For each file for the supported data format
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
def test_data_loader_kwargs(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict,
        cls: BaseDataLoader,
        kwargs: dict
) -> None:
    """This test fails if the kwargs are not properly defined."""

    # For each supported data format in DataLoaderManager
    for data_format, data_loader in data_loader_manager.data_loaders.items():
        # If parametrized class is equal to the corresponding data loader
        if cls == data_loader:
            # Retrieve paths from fixture
            paths = data_paths_dict[data_format]

            # For each file for the supported data format
            for path in paths:
                assert len(data_loader().load(path=path, **kwargs)) == 10
