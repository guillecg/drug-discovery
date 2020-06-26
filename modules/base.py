import pandas as pd

from abc import ABCMeta, abstractmethod


class BaseDataLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(self, path: str, filters: dict = None, **kwargs) -> pd.DataFrame:
        raise NotImplementedError
