#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`core.processors.preprocess.z_score` [module]

Classes
-------
`ZScorer`
"""
# Disable error codes for attributes which are not detected by the type checker:
# (configuration and data attributes are initialized by the base class constructor)
# pylint: disable=useless-parent-delegation

from typing import Union, Any, Tuple, Dict, Optional

import numpy as np

from core.processors.base_processor import Processor


class ZScorer(Processor):
    """
    Z-score data in of samples, with either the standard method or a custom method.

    z-score formula:

    .. math:

        Z = \\frac{x - \\mu}{\\sigma}

    with:

    - :math:`Z`	: standard score for one sample
    - :math:`x` : observed value for one sample
    - :math:`\\mu` :	baseline from which samples deviate
    - :math:`\\sigma` : factor to scale data relative to a variability level

    In the standard z-score method:

    - :math:`\\mu` is the mean across the samples
    - :math:`\\sigma` is the standard deviation of the samples

    In the custom z-score method, the baseline and scaling factor can be different from those of the
    samples set. For instance, they can be computed in a reference condition.

    Class Attributes
    ----------------

    Configuration Attributes
    ------------------------
    axes : Union[int, Tuple[int, ...]], optional
        Axes along which independent mean and standard deviations are computed, if `mu` and `sigma`
        are not provided.
        If not provided, the mean and standard deviations are computed from all the values across
        all the dimensions of `x`.
        If the input parameter is a single integer, it is converted to a one-element tuple for
        consistency.
    mu : Union[float, np.ndarray], optional
        Custom baseline which is subtracted to each sample.
        Shape: ???
    sigma : Union[float, np.ndarray], optional
        Custom scaling factor which divides the centered data.
        Shape: ???

    Processing Arguments
    --------------------
    x : np.ndarray
        Data samples to z-score. Shape: Any, as long as it is compatible with the configuration
        parameters `axes`, `mu` and `sigma` (if provided).
        .. _x:

    Returns
    -------
    z : np.ndarray
        Z-scored data. Shape: Identical to the shape of `x`.

    Methods
    -------
    `z_score`

    Examples
    --------
    Z-score a uni-dimensional set of samples based on global mean and std.
    Here, :math:`\\mu = 1.5` and :math:`\\sigma = 0.5`.

    >>> x = np.array([1, 2, 1, 2, 1, 2])
    >>> zscorer = ZScorer()
    >>> z = zscorer.process(x=x)
    >>> print(z)
    array([-1.,  1., -1.,  1., -1.,  1.])

    Z-score a 2-D set of samples along rows (axis 1).
    Here, :math:`\\mu = [1.5, 0.5]` and :math:`\\sigma = [0.5, 0.5]`.

    >>> x = np.array([[1, 2, 1, 2],
    ...               [0, 1, 0, 1]])
    >>> zscorer = ZScorer(axes=1)
    >>> z = zscorer.process(x=x)
    >>> print(z)
    array([[ 1.,  1.,  1.,  1.],
           [-1.,  1., -1.,  1.]])

    Z-score a 2-D set of samples with a custom mean and std.

    >>> x = np.array([[1, 2, 1, 2],
    ...               [0, 1, 0, 1]])
    >>> mu = np.array([2, 3])
    >>> sigma = np.array([1, 2])
    >>> zscorer = ZScorer(mu=mu, sigma=sigma)
    >>> z = zscorer.process(x=x)
    >>> print(z)
    array([[-1., -0.5, -1., -0.5],
              [-1., -1., -1., -1.]])

    See Also
    --------
    :class:`core.processors.preprocess.base_processor.Processor`
        Base class for all processors: see class-level attributes and template methods.
    """

    IS_RANDOM: bool = False

    def __init__(
        self,
        axes: Optional[Union[int, Tuple[int, ...]]] = None,
        mu: Optional[Union[float, np.ndarray]] = None,
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.mu = mu
        self.sigma = sigma

    # --- Validation Methods -----------------------------------------------------------------------

    def _pre_process(self, **input_data: Any) -> Dict[str, Any]:
        """
        Validate the data with respects to the configuration parameters.

        Raises
        ------
        ValueError
            If the shape of `x` does not contain the axes specified in the configuration parameter
            `axes`.
            If the shape of `x` is incompatible to broadcast with `mu` and `sigma`.

        See Also
        --------
        :func:`np.broadcast_shapes`: Check if two shapes are broadcast-compatible.
        """
        x = input_data.get("x", np.zeros((0,)))
        # Check if axes are within the range of x's dimensions
        if self.axes is not None and any(ax >= x.ndim for ax in self.axes):
            raise ValueError(f"Invalid shape: {x.shape} incompatible with `axes` {self.axes}")
        # Check broadcast compatibility
        for stat, name in zip([self.mu, self.sigma], ["mu", "sigma"]):
            if stat is not None:
                shape = np.shape(stat)  # handle the case of a single number
                if not np.broadcast_shapes(x.shape, shape):
                    raise ValueError(f"Shape of `x` {x.shape} incompatible with `{name}` {shape}.")
        return input_data

    # --- Processing Methods -----------------------------------------------------------------------

    def _process(self, **input_data: Any) -> np.ndarray:
        """Implement the template method called in the base class `process` method."""
        x = input_data["x"]
        mu = self.compute_mean(x) if self.mu is None else self.mu
        sigma = self.compute_std(x) if self.sigma is None else self.sigma
        z = self.z_score(x, mu, sigma)
        return z

    def compute_mean(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the mean of a set of samples.

        Arguments
        ---------
        x : np.ndarray
            See the processing attribute :ref:`x`.

        Returns
        -------
        mu : Union[float, np.ndarray]
            Mean across the samples.
            If `axes` is set, the mean is taken along the specified axes and an array is returned.
            Its shape matches the shape of `x` with the specified `axes` removed.
            Otherwise, the mean is taken across all the values, and a scalar is returned.
        """
        if self.axes is not None:
            mu = np.mean(x, axis=self.axes, keepdims=True)
        else:
            mu = np.mean(x)
        return mu

    def compute_std(self, x: np.ndarray) -> Union[float, np.ndarray]:
        """
        Compute the standard deviation of a set of samples.

        Arguments
        ---------
        x : np.ndarray
            See the processing attribute :ref:`x`.

        Returns
        -------
        sigma : Union[float, np.ndarray]
            STD across the samples. See the `compute_mean` method for details.
        """
        if self.axes is not None:
            sigma = np.std(x, axis=self.axes, keepdims=True)
        else:
            sigma = np.std(x)
        return sigma

    def z_score(self, x, mu, sigma):
        """
        Compute the z-score for the set of samples.

        Arguments
        ---------
        x, mu, sigma : np.ndarray
            See the attributes :ref:`x`, `mu`, `sigma`.

        Raises
        ------
        ZeroDivisionError
            If the array `sigma` contains null values.
        """
        if np.any(np.isclose(sigma, 0)):
            raise ZeroDivisionError("Z-scoring failure: null values in `sigma`")
        z = (x - mu) / sigma
        return z
