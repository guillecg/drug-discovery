import pytest

from rdkit import Chem

from modules.data.data_loaders import DataLoaderManager
from modules.preprocessing.descriptors import *
from modules.utils import load_yaml, get_test_data_paths


# IMPORTANT NOTE: all test files contain, by default, the 10 first records in
# 'data/Tox21/tox21_10k_data_all.sdf'
CONFIG = load_yaml(path='config.yaml')
TEST_DATA_DIR = CONFIG.get('TEST_DATA_DIR')


# TODO: DescriptorMordred with custom descriptors


@pytest.fixture(scope='module')
def data_loader_manager():
    return DataLoaderManager()


@pytest.mark.parametrize(
    'cls, kwargs', [
        (DescriptorMordred, {'ignore_3D': True}),
        pytest.param(DescriptorMordred, {'NaN': 0}, marks=pytest.mark.xfail),
    ]
)
def test_data_loader_kwargs(cls, kwargs):
    """This test fails if the kwargs are not properly defined"""
    assert cls(**kwargs)


@pytest.mark.slowest
@pytest.mark.parametrize(
    'calculator', [
        DescriptorMordred
    ]
)
def test_calculation_individually(data_loader_manager, calculator):
    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

        for path in paths:
            # Load data
            data = data_loader_manager.load(path=path)

            # Generate molecules from SMILES, since not all formats have them
            data['Molecule'] = [Chem.MolFromSmiles(smiles)
                                for smiles in data['SMILES']]

            # Create pipeline and calculate descriptors
            desc_calculator = calculator()
            desc = desc_calculator.fit_transform(X=data['Molecule'].to_numpy())

            assert isinstance(desc, pd.DataFrame)
            assert len(desc) == 10


@pytest.mark.slowest
@pytest.mark.parametrize(
    'steps', [
        [('Mordred', DescriptorMordred())],
        [('Mordred', DescriptorMordred()), ('Mordred', DescriptorMordred())]
    ]
)
def test_calculation_with_pipeline(data_loader_manager, steps):
    for data_type in data_loader_manager.data_loaders.keys():
        paths = get_test_data_paths(
            data_dir=TEST_DATA_DIR,
            data_type=data_type
        )

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
