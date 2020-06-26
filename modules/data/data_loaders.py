import os

import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.base import BaseDataLoader


class DataLoaderManager(BaseDataLoader):

    def __init__(self) -> None:
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


class DataLoaderCSV(BaseDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=path, **kwargs)


class DataLoaderExcel(BaseDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return pd.read_excel(io=path, **kwargs)


class DataLoaderSMILES(BaseDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        # Get SMILES instead of molecules since SMILES are going to be
        # present in all the given data formats
        smiles = Chem.SmilesMolSupplier(path, **kwargs)
        smiles = list(map(Chem.MolToSmiles, smiles))

        return pd.DataFrame.from_dict({'SMILES': smiles})


class DataLoaderSDF(BaseDataLoader):

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            **kwargs
        )
