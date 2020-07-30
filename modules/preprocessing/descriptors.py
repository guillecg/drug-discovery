from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

from typing import List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from mordred import Calculator, descriptors as descriptors_mordred


# TODO: https://fiehnlab.ucdavis.edu/staff/kind/chemoinformatics/concepts/descriptors
# TODO: check whether DescriptorPipeline should inherit from sklearn.Pipeline


class DescriptorPipeline(BaseEstimator, TransformerMixin):
    """Pipeline for calculating descriptors.

    This class handles the calculation of different types of descriptors,
    similarly to the steps in a sklearn.Pipeline.

    Currently, only the following descriptors are supported (see references):
        - RDKit [1]
        - Mordred [2]

    Parameters
    ----------
    mol_column : str
        The column containing the molecule in RDKit type.

    steps : list of tuples (step name, step instance)
        The list of steps containing the descriptors calculators. Descriptors
        are added incrementally to original data.

    Attributes
    ----------
    mol_column : str
        The column containing the molecule in RDKit type.

    steps : list of tuples (step name, step instance)
        The list of steps containing the descriptors calculators.

    Examples
    --------
    >>> from modules.data_loaders import DataLoaderManager
    >>> from modules.preprocessing.descriptors import (
    ...     DescriptorPipeline,
    ...     DescriptorMordred
    ... )
    >>> data_loader = DataLoaderManager()
    >>> data = data_loader.load(path='tests/data/test_data.sdf')
    >>> data[['ID', 'SMILES']]
                    ID                                             SMILES
    0  NCGC00178831-03  C[n+]1c2cc(N)ccc2cc2ccc(N)cc21.Nc1ccc2cc3ccc(N...
    1  NCGC00166114-03  O=C([O-])c1ccccc1-c1c2cc(Br)c(=O)c(Br)c-2oc2c(...
    2  NCGC00263563-01  CO[C@H]1CC(O[C@H]2C[C@H]([C@H]3O[C@](C)(O)[C@H...
    3  NCGC00013058-02  CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccccc2)c...
    4  NCGC00167516-01  CC(=O)O.CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=...
    5  NCGC00018301-05                  CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1
    6  NCGC00249897-01                     Cc1ccc([N+](=O)[O-])c2c1O[Hg]2
    7  NCGC00016000-18                         CCN(CC)C(=S)SSC(=S)N(CC)CC
    8  NCGC00181091-01         CCCCCCOc1ccc(C(=N)N(CCCC)CCCC)c2ccccc12.Cl
    9  NCGC00094089-01  COCC(=O)O[C@]1(CCN(C)CCCc2nc3ccccc3[nH]2)CCc2c...
    >>> desc_pipe = DescriptorPipeline(mol_column='Molecule', steps=[
    ...     ('Mordred', DescriptorMordred())
    ... ])

    # NOTE: Mordred warnings do not allow these commands to pass the tests
    # >>> data = desc_pipe.fit_transform(X=data)
    # 100%|██████████| 10/10 [00:02<00:00,  4.74it/s]
    # >>> data[['SMILES', 'MW']]
    #                                               SMILES           MW
    # 0  C[n+]1c2cc(N)ccc2cc2ccc(N)cc21.Nc1ccc2cc3ccc(N...   468.182922
    # 1  O=C([O-])c1ccccc1-c1c2cc(Br)c(=O)c(Br)c-2oc2c(...   687.674410
    # 2  CO[C@H]1CC(O[C@H]2C[C@H]([C@H]3O[C@](C)(O)[C@H...   933.566100
    # 3  CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccccc2)c...   926.373823
    # 4  CC(=O)O.CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=...  1341.661840
    # 5                  CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1   337.243936
    # 6                     Cc1ccc([N+](=O)[O-])c2c1O[Hg]2   352.997586
    # 7                         CCN(CC)C(=S)SSC(=S)N(CC)CC   296.050933
    # 8         CCCCCCOc1ccc(C(=N)N(CCCC)CCCC)c2ccccc12.Cl   418.275092
    # 9  COCC(=O)O[C@]1(CCN(C)CCCc2nc3ccccc3[nH]2)CCc2c...   567.243076

    References
    ----------
    .. [1] RDKit. http://www.rdkit.org/
    .. [2] H. Moriwaki, Y.-S. Tian, N. Kawashita and T. Takagi,
           "Mordred: a molecular descriptor calculator", Journal of
           Cheminformatics, 10:4, 2018. doi: 10.1186/s13321-018-0258-y
    """

    def __init__(self, mol_column: str, steps: List[tuple]) -> None:
        self.mol_column = mol_column
        self.steps = steps

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> DescriptorPipeline:
        """Placeholder method to follow sklearn.base.TransformerMixin
        architecture: this method is required by the :meth:`fit_transform`
        method.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate descriptors for given data. Note that this method differs
        from the other descriptors calculators since it receives the original
        data as a pandas.DataFrame and not just the array of molecules.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe with the original data with a column containing the
            rdkit.Chem.rdchem.Mol for calculating descriptors.

        Returns
        -------
        X : pandas.DataFrame
            Dataframe with both the original data and calculated descriptors.
        """
        # Reset index to avoid interference while concatenating
        X = X.reset_index(drop=True)

        # Keep track of the original number of rows for later concatenation
        original_n_rows = X.shape[0]

        # For each step in the pipeline
        for _, step_instance in self.steps:
            # Calculate descriptors
            descriptors = step_instance.transform(
                X=X[self.mol_column].to_numpy()
            )

            # Concatenate descriptors as columns to original data
            X = pd.concat(
                [X, descriptors],
                axis=1
            )

        # Avoid errors while concatenating columns with different indexes
        assert original_n_rows == X.shape[0]

        return X


class DescriptorRDKit(BaseEstimator, TransformerMixin):
    """Descriptor calculator for RDKit descriptors.

    References
    ----------
    .. [1] RDKit. http://www.rdkit.org/
    """

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> DescriptorRDKit:
        """Placeholder method to follow sklearn.base.TransformerMixin
        architecture: this method is required by the :meth:`fit_transform`
        method.
        """
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate descriptors for given data.

        Parameters
        ----------
        X : numpy.ndarray
            Array of rdkit.Chem.rdchem.Mol whose descriptors will be
            calculated.

        Returns
        -------
        descriptors : pandas.DataFrame
            Calculated descriptors for the given molecules.
        """
        raise NotImplementedError


class DescriptorMordred(BaseEstimator, TransformerMixin):
    """Descriptor calculator for Mordred descriptors.

    Parameters
    ----------
    descriptors_list : mordred.descriptors
        Mordred descriptors objects which will be used to calculate them.

    Attributes
    ----------
    descriptors_list : mordred.descriptors
        Mordred descriptors objects which will be used to calculate them. It
        is needed as an attribute for avoiding FutureWarnings from
        sklearn.base (see https://stackoverflow.com/a/61396576).

    calculator : mordred._base.calculator.Calculator
        Mordred calculator object that handles the calculation of all passed
        descriptors.

    Examples
    --------
    >>> from modules.data_loaders import DataLoaderManager
    >>> from modules.preprocessing.descriptors import (
    ...     DescriptorPipeline,
    ...     DescriptorMordred
    ... )
    >>> data_loader = DataLoaderManager()
    >>> data = data_loader.load(path='tests/data/test_data.sdf')
    >>> data[['ID', 'SMILES']]
                    ID                                             SMILES
    0  NCGC00178831-03  C[n+]1c2cc(N)ccc2cc2ccc(N)cc21.Nc1ccc2cc3ccc(N...
    1  NCGC00166114-03  O=C([O-])c1ccccc1-c1c2cc(Br)c(=O)c(Br)c-2oc2c(...
    2  NCGC00263563-01  CO[C@H]1CC(O[C@H]2C[C@H]([C@H]3O[C@](C)(O)[C@H...
    3  NCGC00013058-02  CN(C)c1ccc(C(=C2C=CC(=[N+](C)C)C=C2)c2ccccc2)c...
    4  NCGC00167516-01  CC(=O)O.CCNC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=...
    5  NCGC00018301-05                  CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1
    6  NCGC00249897-01                     Cc1ccc([N+](=O)[O-])c2c1O[Hg]2
    7  NCGC00016000-18                         CCN(CC)C(=S)SSC(=S)N(CC)CC
    8  NCGC00181091-01         CCCCCCOc1ccc(C(=N)N(CCCC)CCCC)c2ccccc12.Cl
    9  NCGC00094089-01  COCC(=O)O[C@]1(CCN(C)CCCc2nc3ccccc3[nH]2)CCc2c...
    >>> desc_calc = DescriptorMordred()

    # NOTE: Mordred warnings do not allow these commands to pass the tests
    # >>> desc = desc_calc.fit_transform(X=data['Molecule'].to_numpy())
    # 100%|██████████| 10/10 [00:02<00:00,  4.74it/s]
    # >>> desc[['MW', 'SLogP']]
    #             MW    SLogP
    # 0   468.182922  1.53830
    # 1   687.674410 -0.94010
    # 2   933.566100  3.35840
    # 3   926.373823  3.58460
    # 4  1341.661840 -0.45963
    # 5   337.243936  5.55910
    # 6   352.997586  0.91842
    # 7   296.050933  3.62120
    # 8   418.275092  7.44827
    # 9   567.243076  6.11450

    References
    ----------
    .. [2] H. Moriwaki, Y.-S. Tian, N. Kawashita and T. Takagi,
           "Mordred: a molecular descriptor calculator", Journal of
           Cheminformatics, 10:4, 2018. doi: 10.1186/s13321-018-0258-y
    """

    def __init__(self, descriptors_list=descriptors_mordred, **kwargs) -> None:
        self.descriptors_list = descriptors_list
        self.calculator = Calculator(descriptors_list, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> DescriptorMordred:
        """Placeholder method to follow sklearn.base.TransformerMixin
        architecture: this method is required by the :meth:`fit_transform`
        method.
        """
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        """Calculate descriptors for given data.

        Parameters
        ----------
        X : numpy.ndarray
            Array of rdkit.Chem.rdchem.Mol whose descriptors will be
            calculated.

        Returns
        -------
        descriptors : pandas.DataFrame
            Calculated descriptors for the given molecules.
        """
        return self.calculator.pandas(mols=X)
