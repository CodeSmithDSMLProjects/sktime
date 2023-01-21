# -*- coding: utf-8 -*-
"""
Implements Christiano Fitzgerald bandpass filter transformation.

Please see the original library
(https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/filters/cf_filter.py)
"""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["klam-data", "pyyim", "mgorlin"]
__all__ = ["CFFilter"]


import numpy as np
import pandas as pd
from statsmodels.tools.validation import PandasWrapper, array_like

from sktime.transformations.base import BaseTransformer


class CFFilter(BaseTransformer):
    """Filter a times series using the Christiano Fitzgerald filter.

    This is a wrapper around statsmodels' cffilter function
    (see 'sm.tsa.filters.cf_filter.cffilter').

    Parameters
    ----------
    x : array_like
        The 1 or 2d array to filter. If 2d, variables are assumed to be in
        columns.
    low : float
        Minimum period of oscillations. Features below low periodicity are
        filtered out. Default is 6 for quarterly data, giving a 1.5 year
        periodicity.
    high : float
        Maximum period of oscillations. Features above high periodicity are
        filtered out. Default is 32 for quarterly data, giving an 8 year
        periodicity.
    drift : bool
        Whether or not to remove a trend from the data. The trend is estimated
        as np.arange(nobs)*(x[-1] - x[0])/(len(x)-1).

    Returns
    -------
    cycle : array_like
        The features of x between the periodicities low and high.
    trend : array_like
        The trend in the data with the cycles removed.

    Examples
    --------
    >>> from sktime.transformations.series.cffilter import CFFilter
    >>> import pandas as pd
    >>> import statsmodels.api as sm
    >>> dta = sm.datasets.macrodata.load_pandas().data
    >>> index = pd.date_range(start='1959Q1', end='2009Q4', freq='Q')
    >>> dta.set_index(index, inplace=True)
    >>> cf = CFFilter(6, 32)
    >>> cf_cycles, cf_trend = sm.tsa.filters.cffilter(dta[["infl", "unemp"]])
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Panel",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": [
            pd.RangeIndex
        ],  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "handles-missing-data": False,  # can estimator handle missing data?
        "remember_data": False,  # whether all data seen is remembered as self._X
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        low=6,
        high=32,
        drift=True,
    ):
        self.low = low
        self.high = high
        self.drift = drift
        super(CFFilter, self).__init__()

    def transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : array_like
        A 1 or 2d ndarray. If 2d, variables are assumed to be in columns.

        Returns
        -------
        transformed cyclical version of X
        """
        if self.low < 2:
            raise ValueError("low must be >= 2")
        pw = PandasWrapper(X)
        # df = pd.DataFrame(X.copy())
        X = array_like(X, "X", ndim=2)
        nobs, nseries = X.shape
        a = 2 * np.pi / self.high
        b = 2 * np.pi / self.low

        if self.drift:  # get drift adjusted series
            X = X - np.arange(nobs)[:, None] * (X[-1] - X[0]) / (nobs - 1)

        J = np.arange(1, nobs + 1)
        Bj = (np.sin(b * J) - np.sin(a * J)) / (np.pi * J)
        B0 = (b - a) / np.pi
        Bj = np.r_[B0, Bj][:, None]
        y = np.zeros((nobs, nseries))

        for i in range(nobs):
            B = -0.5 * Bj[0] - np.sum(Bj[1 : -i - 2])
            A = -Bj[0] - np.sum(Bj[1 : -i - 2]) - np.sum(Bj[1:i]) - B
            y[i] = (
                Bj[0] * X[i]
                + np.dot(Bj[1 : -i - 2].T, X[i + 1 : -1])
                + B * X[-1]
                + np.dot(Bj[1:i].T, X[1:i][::-1])
                + A * X[0]
            )
        y = y.squeeze()

        cycle, trend = y.squeeze(), X.squeeze() - y

        # cycle_cols = []
        # for i in df.columns:
        #     cycle_cols.append(str(i)+'_cycle')

        # trend_cols = []
        # for i in df.columns:
        #     trend_cols.append(str(i)+'_trend')

        # cycle = pd.DataFrame(cycle, columns=cycle_cols, index=df.index)
        # trend = pd.DataFrame(trend, columns=trend_cols, index=df.index)

        # return cycle, trend
        # return pd.DataFrame(cycle, columns=cycle_cols, index=df.index),
        # pd.DataFrame(trend, columns=trend_cols, index=df.index)

        return pw.wrap(cycle, append="cycle"), pw.wrap(trend, append="trend")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"low": 6, "high": 32, "drift": True}
        return params
