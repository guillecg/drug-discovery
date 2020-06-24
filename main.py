from modules.data.data_loaders import DataLoaderManager

from modules.preprocessing.pipelines import Pipeline
from modules.preprocessing.smiles import (
    SMILESChecker,
    SMILESEmbedder
)


if __name__ == '__main__':
    # 1. Load data (class DataLoaderManager)
    data_loader = DataLoaderManager()
    data = data_loader.load(path='data/Tox21/tox21_10k_data_all.sdf')

    # 1.1. Data enrichment (class DataEnricher)

    # 2. Preprocess data (class Pipeline)
    # 2.1. Check and sanitize data + EDA
    # 2.2. Feature engineering + EDA
    # 2.3. Feature selection + EDA

    # 3. Fit model (class ModelSelector)
    # 3.1. Hyperparams selection
    # 3.2. Validation strategy (CV, NestedCV, etc.)

    # 4. Evaluate model (class ModelEvaluator)
    # 4.1. Metrics
    # 4.2. Visualization of predictions, generated data, etc.
    # 4.3. Explainability
