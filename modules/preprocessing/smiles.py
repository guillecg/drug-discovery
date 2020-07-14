from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from rdkit import Chem


class SMILESChecker(BaseEstimator, TransformerMixin):
    """Simple RDKit-based SMILES checker.

    As documented in the RDKit documentation, [1] the :param:`canonical`
    parameter in :func:`rdkit.Chem.rdmolfiles.MolToSmiles` function that
    allows to obtain the canonical SMILES.

    References
    ----------
    .. [1] RDKit. https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html
    """

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> SMILESChecker:
        """Placeholder method to follow sklearn.base.TransformerMixin
        architecture: this method is required by the :meth:`fit_transform`
        method.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Sanitize the given SMILES strings.

        Parameters
        ----------
        X : numpy.ndarray
            Array of SMILES strings to be sanitized.

        Returns
        -------
        X : numpy.ndarray
            Array of sanitized SMILES strings.
        """
        X = map(Chem.MolFromSmiles, X)
        X = map(lambda smi: Chem.MolToSmiles(smi, canonical=True), X)

        return np.array(list(X))
