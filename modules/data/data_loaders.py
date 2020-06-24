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

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        file_type = os.path.splitext(path)[-1]

        data_loader = self.data_loaders[file_type]()

        return data_loader.load(path=path, **kwargs)


class DataLoaderCSV(IDataLoader):

    def load(self, path: str, filters: dict = {}) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=path)

        if filters:
            for key, value in filters.items():
                data = data[data[key].isin(list(value))]

        return data


class DataLoaderExcel(IDataLoader):

    def load(self, path: str, filters: dict = {}) -> pd.DataFrame:
        data = pd.read_excel(io=path)

        if filters:
            for key, value in filters.items():
                data = data[data[key].isin(list(value))]

        return data


class DataLoaderSMILES(IDataLoader):

    def load(self, path: str) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {'Molecule': Chem.SmilesMolSupplier(path)}
        )


class DataLoaderSDF(IDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            **kwargs
        )
