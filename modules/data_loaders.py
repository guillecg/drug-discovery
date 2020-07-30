import os

import pandas as pd

from rdkit import Chem
from rdkit.Chem import PandasTools

from modules.base import BaseDataLoader


class DataLoaderManager(BaseDataLoader):
    """Data loaders manager.

    This class handles the loading of different files formats and exposes the
    :meth:`load` for easily loading a given data file path.

    Currently, only the following formats are supported:
        - CSV (.csv)
        - Excel (.xls, .xlsx)
        - SMILES (.smi)
        - SDF (.sdf)

    Attributes
    ----------
    data_loaders : dict
        The supported formats mapped to their corresponding data loaders.

    Examples
    --------
    >>> from modules.data_loaders import DataLoaderManager
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
    >>> data = data_loader.load(
    ...     path='tests/data/test_data.sdf',
    ...     filters={'SMILES': 'CCN(CC)C(=S)SSC(=S)N(CC)CC'}
    ... )
    >>> data['SMILES']
    7    CCN(CC)C(=S)SSC(=S)N(CC)CC
    Name: SMILES, dtype: object
    >>> data = data_loader.load(
    ...     path='tests/data/test_data.sdf',
    ...     filters={'SMILES': [
    ...         'CCN(CC)C(=S)SSC(=S)N(CC)CC',
    ...         'CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1'
    ...     ]}
    ... )
    >>> data['SMILES']
    5    CCCCCCCCNC(C)C(O)c1ccc(SC(C)C)cc1
    7           CCN(CC)C(=S)SSC(=S)N(CC)CC
    Name: SMILES, dtype: object
    """

    def __init__(self) -> None:
        self.data_loaders = {
            '.csv': DataLoaderCSV,
            '.xls': DataLoaderExcel,
            '.xlsx': DataLoaderExcel,
            '.smi': DataLoaderSMILES,
            '.sdf': DataLoaderSDF,
        }

    def load(self, path: str, filters: dict = None, **kwargs) -> pd.DataFrame:
        """Load data from the given file path and, optionally, filter the
        records.

        Parameters
        ----------
        path : str
            The path pointing to the data file.

        filters : dict, default=None
            The filters to apply to the data, specifying the key as the column
            and the desired value as a single element or iterable. It is
            important to highlight that these filters are applied according to
            the internal ordering of the dictionary's items.

        **kwargs : kwargs
            Parameters passed to the different parsers in each :meth:`load`.

        Returns
        -------
        data : pandas.DataFrame
            The contents of the data file.
        """
        # Extract file type from data path
        file_type = os.path.splitext(path)[-1]

        # Choose data loader according to file type
        data_loader = self.data_loaders[file_type]()

        # Load data
        data = data_loader.load(path=path, **kwargs)

        # Filter data if required
        if filters:
            for key, value in filters.items():
                # Convert value to list if specified as a single element in
                # order to be able to apply the series method isin
                value = [value] if not isinstance(value, list) else value

                data = data[data[key].isin(value)]

        return data


class DataLoaderCSV(BaseDataLoader):
    """Data loader for CSV (.csv) files."""

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load data from the given file path.

        Parameters
        ----------
        path : str
            The path pointing to the data file.

        **kwargs : kwargs
            Parameters passed to the CSV parser.

        Returns
        -------
        data : pandas.DataFrame
            The contents of the data file.
        """
        return pd.read_csv(filepath_or_buffer=path, **kwargs)


class DataLoaderExcel(BaseDataLoader):
    """Data loader for Excel (.xls, .xlsx) files."""

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load data from the given file path.

        Parameters
        ----------
        path : str
            The path pointing to the data file.

        **kwargs : kwargs
            Parameters passed to the Excel parser.

        Returns
        -------
        data : pandas.DataFrame
            The contents of the data file.
        """
        return pd.read_excel(io=path, **kwargs)


class DataLoaderSMILES(BaseDataLoader):
    """Data loader for SMILES (.smi) files."""

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load data from the given file path.

        Parameters
        ----------
        path : str
            The path pointing to the data file.

        **kwargs : kwargs
            Parameters passed to the SMILES parser.

        Returns
        -------
        data : pandas.DataFrame
            The contents of the data file.
        """
        # Get SMILES instead of molecules since SMILES are going to be
        # present in all the given data formats
        smiles = Chem.SmilesMolSupplier(path, **kwargs)
        smiles = list(map(Chem.MolToSmiles, smiles))

        return pd.DataFrame.from_dict({'SMILES': smiles})


class DataLoaderSDF(BaseDataLoader):
    """Data loader for SDF (.sdf) files."""

    def load(self, path: str, **kwargs) -> pd.DataFrame:
        """Load data from the given file path.

        Parameters
        ----------
        path : str
            The path pointing to the data file.

        **kwargs : kwargs
            Parameters passed to the SDF parser.

        Returns
        -------
        data : pandas.DataFrame
            The contents of the data file.
        """
        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            **kwargs
        )
