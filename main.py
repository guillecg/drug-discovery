from modules.data.data_loader import DataLoaderCSV

from modules.preprocessing.pipelines import Pipeline
from modules.preprocessing.smiles import (
    SMILESHotEncoder,
    SMILESChecker,
    SMILESEmbedder
)



if __name__ == '__main__':
    # 1. Load data (class DataLoader)
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
