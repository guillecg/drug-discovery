![Python package](https://github.com/guillecg/drug-discovery/workflows/Python%20package/badge.svg)

# Machine Learning for Drug Discovery

This repository aims to provide a modular architecture to rapidly build pipelines that allow the user to discover or repurpose drugs.

## Table of contents
* [Setup](#setup)
* [Getting started](#getting-started)
* [Roadmap](#roadmap)

## Setup
All the dependencies are detailed in the [environment.yml](https://github.com/guillecg/drug-discovery/blob/master/environment.yml) file. To install them, create a new conda environment using that file:
```bash
$ conda env create --name dd --file environment.yml
$ conda activate dd
```

## Getting started
```python
import plotly.express as px

from sklearn.pipeline import Pipeline

from rdkit import Chem

from modules.data_loaders import DataLoaderManager
from modules.preprocessing.smiles import SMILESChecker
from modules.preprocessing.descriptors import (
    DescriptorPipeline,
    DescriptorMordred
)

# Load data
data_loader = DataLoaderManager()
data = data_loader.load(
    path='tests/data/test_data.sdf',
    removeHs=False
)

# Preprocess data (sanitize SMILES)
smiles_pipe = Pipeline(steps=[
    ('SMILESChecker', SMILESChecker())
])
data['SMILES'] = smiles_pipe.fit_transform(
    X=data['SMILES'].to_numpy()
)

# Recalculate mol from curated SMILES
data['Molecule'] = [Chem.MolFromSmiles(smiles)
                    for smiles in data['SMILES']]

# Calculate descriptors
desc_pipe = DescriptorPipeline(mol_column='Molecule', steps=[
    ('Mordred', DescriptorMordred())
])
data = desc_pipe.fit_transform(X=data)

# Visualize descriptors
variables = {
    'x': 'MW',
    'y': 'nHetero',
    'z': 'SLogP',
    'color': 'SR-p53'
}
fig = px.scatter_3d(
    data_frame=data,
    x=variables['x'],
    y=variables['y'],
    z=variables['z'],
    color=variables['color'],
    template='plotly_white',
    height=750,
    width=900,
    title='Initial EDA'
)
fig.show()
```

For detailed examples, please see the [examples](https://github.com/guillecg/drug-discovery/tree/master/examples) folder.

## Roadmap

A detailed roadmap with future lines of work can be found [here](https://github.com/guillecg/drug-discovery/projects/1). Ideas and possible future implementations can also be found [here](https://github.com/guillecg/drug-discovery/blob/master/IDEAS.md).
