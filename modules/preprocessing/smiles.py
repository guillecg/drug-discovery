import numpy as np

from modules.base.interfaces import ITransformer


class SMILESHotEncoder(ITransformer):

    def transform(self, X: np.array) -> np.array:
        raise NotImplementedError


class SMILESChecker(ITransformer):

    def transform(self, X: np.array) -> np.array:
        raise NotImplementedError


class SMILESEmbedder(ITransformer):

    def transform(self, X: np.array) -> np.array:
        raise NotImplementedError
