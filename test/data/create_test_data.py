from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.data_loaders import DataLoaderManager


def create_test_data(path: str, n_rows: int = 10) -> None:
    """Create test data files for all `DataLoaderManager` supported formats
    from a given data file. The resulting files only contain the first `n_rows`
    to avoid overhead in tests.

    Parameters
    ----------
    path : str
        Path to the source file from which all test data files will be
        generated.

    n_rows : int, default=10
        Number of maximum rows to retrieve in order to avoid overhead in tests.
    """

    data_loader = DataLoaderManager()
    data = data_loader.load(path=path)

    # Get only the first `n_rows` records to avoid overhead in tests
    data = data[:n_rows]

    # CSV
    path = 'test/data/test_data.csv'
    data.to_csv(
        path,
        header=True,
        index=False
    )
    assert len(data_loader.load(path=path)) == 10

    # Excel (.xls)
    path = 'test/data/test_data.xls'
    data.to_excel(
        path,
        header=True,
        index=False
    )
    assert len(data_loader.load(path=path)) == 10

    # Excel (.xlsx)
    path = 'test/data/test_data.xlsx'
    data.to_excel(
        path,
        header=True,
        index=False
    )
    assert len(data_loader.load(path=path)) == 10

    # SMILES
    path = 'test/data/test_data.smi'
    w = Chem.SmilesWriter(path)
    for smi in data['Molecule']:
        w.write(smi)
    w.close()
    assert len(data_loader.load(path=path)) == 10

    # SDF
    path = 'test/data/test_data.sdf'
    PandasTools.WriteSDF(
        data,
        path,
        molColName='Molecule',
        properties=data.columns.tolist()
    )
    assert len(data_loader.load(path=path)) == 10


if __name__ == '__main__':
    create_test_data(path='data/Tox21/tox21_10k_data_all.sdf')
