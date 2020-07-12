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
    >>> data = data_loader.load(path='data/Tox21/tox21_10k_data_all.sdf')
    >>> data
                 Formula  ... NR-AR-LBD
    0         C27H25ClN6  ...       NaN
    1      C20H6Br4Na2O5  ...       NaN
    2         C47H83NO17  ...       NaN
    3        C52H54N4O12  ...       NaN
    4       C66H87N17O14  ...       NaN
                  ...  ...       ...
    11759      C7H10N2OS  ...         0
    11760        C3H6N2S  ...         0
    11761        C3H6N2S  ...         0
    11762    C10H14NO5PS  ...         0
    11763     C15H26O6S3  ...         0

    [11758 rows x 18 columns]

    >>> data = data_loader.load(
        path='data/Tox21/tox21_10k_data_all.sdf',
        filters={'SMILES': 'COc1cc(C)ccc1O'}
    )
    >>> data['SMILES']
    7814    COc1cc(C)ccc1O
    Name: SMILES, dtype: object

    >>> data = data_loader.load(
        path='data/Tox21/tox21_10k_data_all.sdf',
        filters={'SMILES': ['COc1cc(C)ccc1O', 'ClC(Cl)=C(Cl)Cl']}
    )
    >>> data
    7170    ClC(Cl)=C(Cl)Cl
    7814     COc1cc(C)ccc1O
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

        filters : dict, optional
            The filters to apply to the data, specifying the key as the column
            and the desired value as a single element or iterable. It is
            important to highlight that these filters are applied according to
            the internal ordering of the dictionary's items.

        **kwargs : kwargs
            Parameters passed to the different parsers in each :meth:`load`.

        Returns
        -------
        data : Pandas.DataFrame
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
        data : Pandas.DataFrame
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
        data : Pandas.DataFrame
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
        data : Pandas.DataFrame
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
        data : Pandas.DataFrame
            The contents of the data file.
        """

        return PandasTools.LoadSDF(
            filename=path,
            smilesName='SMILES',
            molColName='Molecule',
            **kwargs
        )
