#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.fit_models.linear_regression` [module]

Classes
-------
LinearRegressionModel

Notes
-----
The processor is not called `LinearRegression` to avoid confusion with the scikit-learn class.

"""
# DISABLED WARNINGS
# --------------------------------------------------------------------------------------------------
# pylint: disable=arguments-differ
# Scope: `process` method in `LinearRegressionModel`.
# Reason: See the note in ``core/__init__.py``
# --------------------------------------------------------------------------------------------------


from typing import Literal, overload, TypeAlias, Any, Tuple, Union, List, Dict

import numpy as np
from sklearn.linear_model import LinearRegression


from core.processors.base_processor import Processor


class LinearRegressionModel(Processor):
    """
    Fitting a linear regression model on input data.
    This class fits a linear model to predict the target variable `y` from predictors `X`.
    The model uses the ordinary least squares method for fitting.

    Configuration Attributes
    ------------------------
    fit_intercept : bool, optional
        Whether to calculate the intercept for this model. If set to False, no intercept will be
        used.
    normalize : bool, optional
        Whether to normalize the predictors before fitting. If `fit_intercept` is set to False, this
        is ignored.

    Processing Arguments
    --------------------
    X : np.ndarray
        Predictor data of shape (n_samples, n_features).
    y : np.ndarray
        Target values of shape (n_samples,).

    Returns
    -------
    coefficients : np.ndarray
        Coefficients of the linear model, shape (n_features,).
    intercept : float
        Intercept of the model, scalar.

    Methods
    -------
    `fit`
    `predict`

    Examples
    --------


    See Also
    --------
    `core.processors.preprocess.base_processor.Processor`
    """

    IS_RANDOM = False

    def __init__(self):
        super().__init__()

    # --- Processing Methods -----------------------------------------------------------------------

    def _process(self, **input_data: Any):
        """Implement the template method called in the base class `process` method."""

    def __init__(self, fit_intercept: bool = True, normalize: bool = False) -> None:
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.is_fitted = False

    # --- Validation Method -----------------------------------------------------------------------

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Validate the input data for linear regression fitting.

        Raises
        ------
        ValueError
            If `X` and `y` have incompatible shapes.
        """
        X = input_data.get("X")
        y = input_data.get("y")

        if X is None or y is None:
            raise ValueError("Both `X` (predictors) and `y` (target) must be provided.")
        if X.ndim != 2:
            raise ValueError("`X` must be a 2D array with shape (n_samples, n_features).")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                "`y` must be a 1D array with length matching the number of samples in `X`."
            )

        return {"X": X, "y": y}

    # --- Processing Method -----------------------------------------------------------------------

    def _process(self, **input_data: Any) -> Dict[str, Union[np.ndarray, float]]:
        """Fits the linear model to the data and returns coefficients and intercept."""
        X = input_data["X"]
        y = input_data["y"]

        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True

        # Retrieve coefficients and intercept
        coefficients = self.model.coef_
        intercept = self.model.intercept_

        return {"coefficients": coefficients, "intercept": intercept}

    # --- Prediction Method -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the fitted linear model.

        Parameters
        ----------
        X : np.ndarray
            Predictor data for which to predict target values.
            Shape: (n_samples, n_features).

        Returns
        -------
        predictions : np.ndarray
            Predicted target values, shape (n_samples,).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.is_fitted:
            raise ValueError(
                "Model has not been fitted yet. Call `_process` with training data first."
            )

        return self.model.predict(X)
