"""
Useful general functionalities
S. Filhol, December 2024


"""

import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self):
        """
        Class to define a z-score scaler with transform and inverse transform functionalities

        zdata = (data - mean(data))/std(data)

        """
        self.mu = None
        self.sigma = None

    def fit(self, data):
        self.mu = np.mean(data)
        self.sigma = np.std(data)

    def transform(self, data):
        return (data - self.mu) / self.sigma

    def inv_transform(self, zdata):
        return zdata * self.sigma + self.mu

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class RobustScaler:
    def __init__(self):
        """
        Class to define a robust z-score scaler with transform and inverse transform functionalities

        zdata = (data - median(data))/std(data)

        """
        self.mu = None
        self.sigma = None

    def fit(self, data):
        self.mu = np.median(data)
        self.sigma = np.std(data)

    def transform(self, data):
        return (data - self.mu) / self.sigma

    def inv_transform(self, zdata):
        return zdata * self.sigma + self.mu

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def get_POT_values(ts, threshold, mtd='5D'):
    """
    Function to extract values from timseries for the GPD analysis

    Args:
        ts (float): time series
        threshold (float): threshold over which values of ts are considered extreme values and considered in GPD processing
        mtd (freq): minimal time distance between 2 max
    Return:
        POT values
    """

    sub = ts[ts >= threshold]
    freq2 = 2 * pd.Timedelta(mtd)
    dd = sub.loc[sub == sub.rolling(freq2, center=True).max()]
    POT_values = dd.loc[(dd.index.to_series().diff() >= pd.Timedelta(mtd)) | np.isnan(dd.index.to_series().diff())]
    return POT_values


def get_BM_values(ts, freq = '365.25D', origin='start'):
    """
	Function to extract Block Maxima values of a timeseries

    Args:
        ts (float): time series
        freq (freq): 
        mtd (freq): minimal time distance between 2 max
    Return:
        BM_values
    """
    def max_with_date(group):
        return group.idxmax()

    date_BM = ts.resample(freq, origin=origin).apply(max_with_date)
    ts_BM = ts[date_BM]
    return ts_BM

