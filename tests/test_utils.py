import pytest

from modules.data_loaders import DataLoaderManager
from modules.utils import load_yaml, get_test_data_paths


@pytest.mark.parametrize('path', ['tests/config.yml'])
def test_load_yaml(path: str) -> None:
    """This tests fails if the loaded file is not a dict."""

    yaml_file = load_yaml(path=path)

    assert isinstance(yaml_file, dict)


def test_get_test_data_paths(
    data_loader_manager: DataLoaderManager,
    config: dict
) -> None:
    """This tests fails if:
        - Returned paths are not a list
        - Any element of the list is not a string
        - No tests file with the required format is found
    """

    # Get tests data directory from tests config
    test_data_dir = config.get('TEST_DATA_DIR')

    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=test_data_dir,
            data_type=data_type
        )

        assert isinstance(paths, list)
        assert all([isinstance(path, str) for path in paths])
        assert len(paths)
