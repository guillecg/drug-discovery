from __future__ import annotations  # See: https://stackoverflow.com/a/33533514

import numpy as np

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn import metrics
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ModelSelector(object):
    """A model selector algorithm which attempts to obtain the best model for a
    given set of parameters and data.

    The selection process consists of two different steps:
        - Finetuning of each model w.r.t. their provided hyperparameters.
        - Selection of the best model according to its performance in tests data
        using the specified scoring function.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration of the ModelSelector instance.
        Please, refer to the sample YAML files in the config folder.
        Accepted YAML keys are:
            - 'ProblemType': For information purposes. String defined by the
            user indicating the type of problem: classification or regression.

            - 'CV': Integer passed to the search algorithm, determines the
            number of folds to use in the hyperparameter tuning. See 'cv' in
            the attributes section.

            - 'ParallelJobs': Integer passed to the search algorithm,
            determines the number of jobs to run in parallel. See 'n_jobs' in
            the attributes section.

            - 'Scoring': String referring to a scorer function that must be
            present in sorted(sklearn.metrics.SCORERS.keys()).

            - 'Hyperparameters': Contains one key per tested algorithm. These
            keys have as value another dictionary containing each
            hyperparameter as key and a list of its possible values (important,
            must be a list even of one element). An example would be:
            {'SVC': {kernel: ['rbf']}}.

    random_search : bool, default=False
        Determines the use of either GridSearchCV or RandomizedSearchCV.

    seed : int, default=42
        This parameter is ignored when random_search is set to False. If True,
        this seed is passed to RandomizedSearchCV as random_state.

    Attributes
    ----------
    models_dict : dict
        Dictionary containing all accepted scikit-learn estimators.

    hyperparameters : dict
        Contains one key per tested algorithm. These keys have as value another
        dictionary containing each hyperparameter as key and a list of its
        possible values (important, must be a list even of one element).
        An example would be: {'SVC': {kernel: ['rbf']}}.

    scoring_fn : callable
        Scoring function retrieved from sklearn.metrics using a string defined
        by the user, which must be in sorted(sklearn.metrics.SCORERS.keys()).
        For more details, refer to sklearn.model_selection.GridSearchCV
        documentation.

    problem_type : str, default='Unknown'
        For information purposes. String defined by the user indicating the
        type of problem: classification or regression.

    cv : int, default=5
        Determines the cross-validation splitting strategy. For more details,
        refer to sklearn.model_selection.GridSearchCV documentation.

    n_jobs : int, default=None
        Number of jobs to run in parallel. For more details, refer to
        sklearn.model_selection.GridSearchCV documentation.

    search_alg : object (sklearn estimator)
        Class used for hyperparameter tuning. If random_search is set to
        True, RandomizedSearchCV will be used; otherwise, GridSearchCV will be
        set as the hyperparameter tuning class.

    seed : int, default=42
        This parameter is ignored when random_search is set to False. If True,
        this seed is passed to RandomizedSearchCV as random_state.

    best_model : object (sklearn estimator), default=None
        Best performing model in tests data according to the specified
        scoring function.

    best_score : float, default=None
        Best score corresponding to the best model.

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from modules.utils import load_yaml

    # Load config and instantiate ModelSelector
    >>> config = load_yaml(
    ...     path='tests/model_selector_config/classification.yml'
    ... )
    >>> model_selector = ModelSelector(config=config)

    # Perform holdout split with ModelSelector's random seed
    >>> X, y = datasets.load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y,
    ...     test_size=0.33,
    ...     random_state=model_selector.seed,
    ...     stratify=y
    ... )

    # Find best model
    >>> model = model_selector.select_model(
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test
    ... )
    """

    models_dict = {
        'SVC': SVC,
        'SVR': SVR,
        'RandomForestClassifier': RandomForestClassifier,
        'RandomForestRegressor': RandomForestRegressor
    }

    def __init__(
        self,
        config: dict,
        random_search: bool = False,
        seed: int = 42
    ) -> None:
        self.hyperparameters = config.get('Hyperparameters')
        self.scoring_fn = metrics.get_scorer(config.get('Scoring'))

        self.problem_type = config.get('ProblemType', 'Unknown')
        self.cv = config.get('CV', 5)
        self.n_jobs = config.get('ParallelJobs', None)

        self.search_alg = RandomizedSearchCV if random_search else GridSearchCV

        self.seed = seed

        self._best_model = None
        self._best_score = None

    @property
    def best_model(self):
        """Auxiliary property for keeping track of the best model."""
        return self._best_model

    @property
    def best_score(self):
        """Auxiliary property for keeping track of the best score corresponding
         to the best model."""
        return self._best_score

    def _perform_search(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Step 1 (on train): Perform hyperparameter search.

        Parameters
        ----------
        X : numpy.ndarray
            Array of training features.

        y : numpy.ndarray
            Array of training labels.

        Returns
        -------
        tuned_models : dict
            Dictionary containing a tuned model (using the corresponding
            search algorithm) for each model name in hyperparameters.
        """
        tuned_models = dict()
        for model_name, model_params in self.hyperparameters.items():
            model_class = self.models_dict.get(model_name)

            # Condition needed to handle different argument names in the two
            # used search classes: param_grid vs param_distributions
            if self.search_alg == RandomizedSearchCV:
                model = self.search_alg(
                    estimator=model_class(),
                    param_distributions=model_params,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    scoring=self.scoring_fn,
                    random_state=self.seed
                )

            elif self.search_alg == GridSearchCV:
                model = self.search_alg(
                    estimator=model_class(),
                    param_grid=model_params,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    scoring=self.scoring_fn
                )

            else:
                raise Exception('Search algorithm is not supported.')

            model.fit(X, y)

            tuned_models[model_name] = model.best_estimator_

        return tuned_models

    def _perform_model_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tuned_models: dict
    ) -> object:
        """Step 2 (on tests): Perform model selection.

        Parameters
        ----------
        X : numpy.ndarray
            Array of tests features.

        y : numpy.ndarray
            Array of tests labels.

        tuned_models : dict
            Dictionary containing a tuned model (using the corresponding
            search algorithm) for each model name in hyperparameters.

        Returns
        -------
        best_model : object (sklearn estimator)
            Best performing model in tests data according to the specified
            scoring function.
        """
        for model_name, model in tuned_models.items():
            # Calculate model score
            model_score = self.scoring_fn(
                estimator=model,
                X=X,
                y_true=y
            )

            # Initialize best model and best score every in the first iteration
            if not self._best_score:
                self._best_model = model
                self._best_score = model_score

            # Update best model and best score if model score is better
            elif model_score > self._best_score:
                self._best_model = model
                self._best_score = model_score

        return self._best_model

    def select_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> object:
        """Main method for selecting the best model according to the provided
        hyperparameters and scoring function.

        The selection process consists of two different steps:
            1. Finetuning of each model w.r.t. their provided hyperparameters.
            2. Selection of the best model according to its performance in
            tests data using the specified scoring function.

        Parameters
        ----------
        X_train : numpy.ndarray
            Array of training features.

        y_train : numpy.ndarray
            Array of training labels.

        X_test : numpy.ndarray
            Array of tests features.

        y_test : numpy.ndarray
            Array of tests labels.

        Returns
        -------
        best_model : object (sklearn estimator)
            Best performing model in tests data according to the specified
            scoring function.
        """
        # Initialize best model and best score every time the method is called
        self._best_model = None
        self._best_score = None

        # Step 1 (on train): Perform hyperparameter search
        tuned_models = self._perform_search(X=X_train, y=y_train)

        # Step 2 (on tests): Perform model selection
        best_model = self._perform_model_selection(
            X=X_test,
            y=y_test,
            tuned_models=tuned_models
        )

        return best_model
