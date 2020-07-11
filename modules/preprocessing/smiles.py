from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from rdkit import Chem


class SMILESChecker(BaseEstimator, TransformerMixin):

    def fit(self, X: np.array, y: np.array = None) -> SMILESChecker:
        return self

    def transform(self, X: np.array, y: np.array = None) -> np.array:
        X = map(Chem.MolFromSmiles, X)
        X = map(Chem.MolToSmiles, X)

        return np.array(list(X))
