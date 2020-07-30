import pytest

from typing import List

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from modules.preprocessing.smiles import SMILESChecker


@pytest.fixture(scope='module')
def smiles_checker():
    return SMILESChecker()


@pytest.mark.slow
def test_smiles_checker(
        smiles_checker: SMILESChecker,
        data_list: List[pd.DataFrame]
) -> None:
    """This tests fails if:
        - Returned SMILES are not a NumPy.ndarray
        - Any element of the array is not a string
        - Length of curated SMILES is not the same as original
    """

    for data in data_list:
        smiles = smiles_checker.fit_transform(
            X=data['SMILES'].to_numpy()
        )

        assert isinstance(smiles, np.ndarray)
        assert all([isinstance(smi, str) for smi in smiles])
        assert len(smiles) == len(data['SMILES'])


@pytest.mark.slow
def test_smiles_checker_with_pipeline(data_list: List[pd.DataFrame]) -> None:
    """This tests fails if:
        - Returned SMILES are not a NumPy.ndarray
        - Any element of the array is not a string
        - Length of curated SMILES is not the same as original
    """

    for data in data_list:
        smiles_pipe = Pipeline(steps=[
            ('SMILESChecker', SMILESChecker())
        ])
        smiles = smiles_pipe.fit_transform(
            X=data['SMILES'].to_numpy()
        )

        assert isinstance(smiles, np.ndarray)
        assert all([isinstance(smi, str) for smi in smiles])
        assert len(smiles) == len(data['SMILES'])
