import pytest

from rdkit import Chem

from modules.data.data_loaders import DataLoaderManager
from modules.preprocessing.descriptors import *


# TODO: DescriptorMordred with custom descriptors


@pytest.mark.parametrize(
    'cls, kwargs', [
        (DescriptorMordred, {'ignore_3D': True}),
        pytest.param(DescriptorMordred, {'NaN': 0}, marks=pytest.mark.xfail),
    ]
)
def test_data_loader_kwargs(cls, kwargs: dict) -> None:
    """This test fails if the kwargs are not properly defined."""
    assert cls(**kwargs)


@pytest.mark.slowest
@pytest.mark.parametrize(
    'calculator', [
        DescriptorMordred
    ]
)
def test_calculation_individually(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict,
        calculator
) -> None:
    """This test fails if:
        - Calculated descriptors are not a Pandas.DataFrame
        - Calculated descriptors length does not match data length
    """

    # For each supported data format in DataLoaderManager
    for data_format, data_loader in data_loader_manager.data_loaders.items():
        # Retrieve paths from fixture
        paths = data_paths_dict[data_format]

        # For each file for the supported data format
        for path in paths:
            # Load data using DataLoaderManager
            data = data_loader_manager.load(path=path)

            # Generate molecules from SMILES, since not all formats have them
            data['Molecule'] = [Chem.MolFromSmiles(smiles)
                                for smiles in data['SMILES']]

            # Create pipeline and calculate descriptors
            desc_calculator = calculator()
            desc = desc_calculator.fit_transform(X=data['Molecule'].to_numpy())

            assert isinstance(desc, pd.DataFrame)
            assert len(desc) == len(data)


@pytest.mark.slowest
@pytest.mark.parametrize(
    'steps', [
        [('Mordred', DescriptorMordred())],
        [('Mordred', DescriptorMordred()), ('Mordred', DescriptorMordred())]
    ]
)
def test_calculation_with_pipeline(
        data_loader_manager: DataLoaderManager,
        data_paths_dict: dict,
        steps: List[tuple]
) -> None:
    """This test fails if:
        - Calculated descriptors are not a Pandas.DataFrame
        - Calculated descriptors length does not match 10 (data test length)
        - Number of resulting columns is NOT higher than in original data
    """

    # For each supported data format in DataLoaderManager
    for data_format, data_loader in data_loader_manager.data_loaders.items():
        # Retrieve paths from fixture
        paths = data_paths_dict[data_format]

        # For each file for the supported data format
        for path in paths:
            # Load data and save original columns for later assessment
            data = data_loader_manager.load(path=path)
            original_columns = data.columns.copy()

            # Generate molecules from SMILES, since not all formats have them
            data['Molecule'] = [Chem.MolFromSmiles(smiles)
                                for smiles in data['SMILES']]

            # Create pipeline and calculate descriptors
            desc_pipe = DescriptorPipeline(mol_column='Molecule', steps=steps)
            data = desc_pipe.fit_transform(X=data)

            assert isinstance(data, pd.DataFrame)
            assert len(data) == 10

            # Assumption: more columns implies that descriptors were calculated
            assert len(data.columns) > len(original_columns)
