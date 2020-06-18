import pandas as pd

from modules.base.interfaces import IDataLoader


class DataLoaderCSV(IDataLoader):

    def load(self, path: str, filters: dict) -> pd.DataFrame:
        raise NotImplementedError


class DataLoaderExcel(IDataLoader):

    def load(self, path: str, filters: dict) -> pd.DataFrame:
        raise NotImplementedError


class DataLoaderDB(IDataLoader):

    def load(self, db, query: str) -> pd.DataFrame:
        raise NotImplementedError
