import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.base.interfaces import IDataLoader


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


class DataLoaderDB(IDataLoader):

    def load(self, db, query: str) -> pd.DataFrame:
        raise NotImplementedError


class DataLoaderSMILES(IDataLoader):

    def load(self, path: str) -> pd.DataFrame:
        return pd.DataFrame.from_dict({'SMILES': Chem.SmilesMolSupplier(path)})


class DataLoaderSDF(IDataLoader):

    def load(self, path: str, removeHs: bool = False) -> pd.DataFrame:
        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            includeFingerprints=True,
            removeHs=removeHs,
            strictParsing=True
        )
