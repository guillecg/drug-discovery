from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

from typing import List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from mordred import Calculator, descriptors as descriptors_mordred


# TODO: https://fiehnlab.ucdavis.edu/staff/kind/chemoinformatics/concepts/descriptors
# TODO: check whether DescriptorPipeline should inherit from sklearn.Pipeline


class DescriptorPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, mol_column: str, steps: List[tuple]) -> None:
        self.mol_column = mol_column
        self.steps = steps

    def fit(self, X: np.array, y: np.array = None) -> DescriptorPipeline:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Reset index to avoid interference while concatenating
        X = X.reset_index(drop=True)

        # Keep track of the original number of rows for later concatenation
        original_n_rows = X.shape[0]

        for _, step_instance in self.steps:
            descriptors = step_instance.transform(
                X=X[self.mol_column].to_numpy()
            )

            X = pd.concat(
                [X, descriptors],
                axis=1
            )

        # Avoid errors while concatenating columns with different indexes
        assert original_n_rows == X.shape[0]

        return X


class DescriptorMordred(BaseEstimator, TransformerMixin):

    def __init__(self, descriptors=descriptors_mordred, **kwargs) -> None:
        self.calculator = Calculator(descriptors, **kwargs)

    def fit(self, X: np.array, y: np.array = None) -> DescriptorMordred:
        return self

    def transform(self, X: np.array) -> pd.DataFrame:
        return self.calculator.pandas(X)


class DescriptorRDKit(BaseEstimator, TransformerMixin):

    def fit(self, X: np.array, y: np.array = None) -> DescriptorRDKit:
        return self

    def transform(self, X: np.array) -> pd.DataFrame:
        raise NotImplementedError
