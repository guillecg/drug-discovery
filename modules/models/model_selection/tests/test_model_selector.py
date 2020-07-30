import pytest

from typing import Tuple

import os

import numpy as np

from modules.models.model_selection import ModelSelector
from modules.utils import load_yaml


@pytest.mark.parametrize(
    'problem_type', ['classification', 'regression']
)
def test_model_selector_creation(
    config: dict,
    problem_type: str
):
    path = config.get('MODEL_SELECTOR_CONFIG_DIR')
    path = os.path.join(path, f'{problem_type}.yml')
    model_selector_config = load_yaml(path=path)

    assert ModelSelector(config=model_selector_config)


@pytest.fixture(scope='module')
def model_selector_classification(config: dict):
    path = config.get('MODEL_SELECTOR_CONFIG_DIR')
    path = os.path.join(path, 'classification.yml')
    model_selector_config = load_yaml(path=path)

    return ModelSelector(config=model_selector_config)


@pytest.fixture(scope='module')
def model_selector_regression(config: dict):
    path = config.get('MODEL_SELECTOR_CONFIG_DIR')
    path = os.path.join(path, 'regression.yml')
    model_selector_config = load_yaml(path=path)

    return ModelSelector(config=model_selector_config)


def test_selection_for_classification(
    model_selector_classification: ModelSelector,
    data_classification: Tuple[np.ndarray]
):
    X_train, X_test, y_train, y_test = data_classification

    model = model_selector_classification.select_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    assert model
    assert model_selector_classification.best_model is not None
    assert model_selector_classification.best_score is not None


def test_selection_for_regression(
    model_selector_regression: ModelSelector,
    data_regression: Tuple[np.ndarray]
):
    X_train, X_test, y_train, y_test = data_regression

    model = model_selector_regression.select_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    assert model
    assert model_selector_regression.best_model is not None
    assert model_selector_regression.best_score is not None
