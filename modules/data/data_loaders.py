import os

import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.base.interfaces import IDataLoader


class DataLoaderManager(IDataLoader):

    def __init__(self):
        self.data_loaders = {
            '.csv': DataLoaderCSV,
            '.xls': DataLoaderExcel,
            '.xlsx': DataLoaderExcel,
            '.smi': DataLoaderSMILES,
            '.sdf': DataLoaderSDF,
        }

    def load(self, path: str, filters: dict = None, **kwargs) -> pd.DataFrame:
        # Extract file type from data path
        file_type = os.path.splitext(path)[-1]

        # Choose data loader according to file type
        data_loader = self.data_loaders[file_type]()

        # Load data
        data = data_loader.load(path=path, **kwargs)

        # Filter data if required
        if filters:
            for key, value in filters.items():
                data = data[data[key].isin(list([value]))]

        return data


class DataLoaderCSV(IDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=path, **kwargs)


class DataLoaderExcel(IDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_excel(io=path, **kwargs)


class DataLoaderSMILES(IDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        smiles = [Chem.MolToSmiles(mol) for mol in
                  Chem.SmilesMolSupplier(path, **kwargs)]

        return pd.DataFrame.from_dict({'SMILES': smiles})


class DataLoaderSDF(IDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            **kwargs
        )
