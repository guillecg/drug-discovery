from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.data.data_loaders import DataLoaderManager


def create_test_data(path: str):

    data_loader = DataLoaderManager()
    data = data_loader.load(path=path)

    # Get only the first 10 records to avoid overhead in tests
    data = data[:10]

    # CSV
    path = 'test/data/test_default.csv'

    data.to_csv(
        path,
        header=True,
        index=False
    )

    assert len(data_loader.load(path=path)) == 10

    # Excel (.xls)
    path = 'test/data/test_default.xls'

    data.to_excel(
        path,
        header=True,
        index=False
    )

    assert len(data_loader.load(path=path)) == 10

    # Excel (.xlsx)
    path = 'test/data/test_default.xlsx'

    data.to_excel(
        path,
        header=True,
        index=False
    )

    assert len(data_loader.load(path=path)) == 10

    # SMILES
    path = 'test/data/test_default.smi'

    w = Chem.SmilesWriter(path)
    for smi in data['Molecule']:
        w.write(smi)
    w.close()

    assert len(data_loader.load(path=path)) == 10

    # SDF
    path = 'test/data/test_default.sdf'

    PandasTools.WriteSDF(
        data,
        path,
        molColName='Molecule',
        properties=data.columns.tolist()
    )

    assert len(data_loader.load(path=path)) == 10


if __name__ == '__main__':
    create_test_data(path='data/Tox21/tox21_10k_data_all.sdf')
