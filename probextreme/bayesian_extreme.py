"""
Functionalities using Bayesian approch for extreme value analysis
Simon Filhol, December 2024
"""

import numpy as np
import arviz as az
import pymc as pm
import pymc_experimental.distributions as pmx
import pytensor.tensor as pt

from matplotlib import pyplot as plt
from . import utils
from . import stat_test as stt


def bayesian_stationary_gev(ts, return_periods=np.array([2,5,10,20,50,100]), return_levels=11):
    """
    Function to fit GEV using the Bayesian approach. This model assumes data stationarity

    Args:
        ts (timeseries, array): data to be fitted by stationary GEV
        return_periods (int array): return period to compute return level for
        return_levels (int, array): return level to compute return period for

    Return:
        model, idata, scaler
    """

    scaler = utils.StandardScaler()
    scaler.fit(ts)
    zdata = scaler.transform(ts)
    p_vec = 1/return_periods
    r_vec = scaler.transform(return_levels)

    with pm.Model(coords={"return_level": r_vec, "probability": p_vec}) as model:
        # Priors
        p = pm.Data('p', p_vec, coords='probability')
        ret = pm.Data('level', r_vec, coords='return_level')

        mu = pm.Normal("mu", mu=0, sigma=0.5)
        sig = pm.HalfNormal("sig", sigma=0.3)
        xi = pm.TruncatedNormal("xi", mu=0, sigma=0.2, lower=-0.6, upper=0.6)

        # Estimation
        gev = pmx.GenExtreme("gev", mu=mu, sigma=sig, xi=xi, observed=zdata)
        # Return levels and return period
        z_p = pm.Deterministic("z_p",
                               mu - sig / xi * (1 - (-pm.math.log(1 - p)) ** (-xi)),
                               dims='probability')
        period = pm.Deterministic("return_period",
                                  1/(1-pm.math.exp(pmx.GenExtreme.logcdf(ret, mu=mu, sigma=sig, xi=xi))),
                                  dims='return_level')


    idata = pm.sample_prior_predictive(samples=1000, model=model)
    with model:
        trace = pm.sample(
            5000,
            cores=4,
            chains=4,
            tune=2000,
            initvals={"mu": -0.5, "sig": 1.0, "xi": -0.1},
            target_accept=0.98,
        )
    # add trace to existing idata object
    idata.extend(trace)
    idata.extend(pm.sample_posterior_predictive(trace, model=model))

    return model, idata, scaler

def model_gpd_linear(zdata):
    print("ERROR: To be implemented")
    pass


def model_gev_linear(zdata):
    """
    A Bayesian GEV model with Loc and Shape being linearly time dependent

    Args:
        zdata (array): standardized data to use for Bayesian inference

    Returns:
        model (pymc model)
    """
    with pm.Model() as model:

        # define time and return period to compute
        t_vec = np.arange(len(zdata))
        p_vec = 1/np.array([2,5,10,20,50,100,200])
        ps_vec = (np.zeros((len(p_vec), len(zdata)))+1).T* p_vec

        # Dimensions
        t = pm.Data('t', t_vec.T, dims='time')
        p = pm.Data('p', ps_vec.T, dims=['probability','time'])

        # Priors
        alpha_mu = pm.Normal("alpha_mu", mu=-0.5, sigma=1)
        beta_mu = pm.Normal("beta_mu", mu=0, sigma=1)
        alpha_sig = pm.Normal("alpha_sig", mu=0, sigma=1)
        beta_sig = pm.Normal("beta_sig", mu=0, sigma=1)

        sig_raw = alpha_sig + beta_sig * t
        sig = pm.Deterministic("sig", pm.math.switch(sig_raw > 0, sig_raw, 0), dims='time')

        mu = pm.Deterministic("mu", alpha_mu + beta_mu * t , dims='time')
        xi = pm.TruncatedNormal("xi", mu=0, sigma=0.2, lower=-0.6, upper=0.6)

        # Estimation
        gev = pmx.GenExtreme("gev", mu=mu, sigma=sig, xi=xi, observed=zdata, dims='time')

        # Return level
        z_p = pm.Deterministic("z_p",  mu - sig / xi * (1 - (-pm.math.log(1 - p)) ** (-xi)),  dims=['probability', 'time'])

    return model


class Bayesian_Extreme:
    """
    Class to perform bayesian modeling of extreme values with by default time dependence.
    This class is wrapper of PyMC method and intends to provide a basic model. For advanced controls on the parameters, refer to PyMC methods on top of which this class is built.

    Attributes:
        ts (timeseries or dataframe): time series of maximum. Default model is GEV so ts must contain block maximum values
        scaler (obj): scaling object. See utils.py

    Methods:
        scale_data()
        assess_stationarity(test=['adfuller', 'ADFuller variance'], freq=30)
        default_gev_model()
        sample_prior(samples=1000)
        infer_posterior(samples=2000)
        evaluate_posterior()

    """
    def __init__(self, ts, scaler=utils.StandardScaler()):
        self.model = None
        self.idata = None
        self.trace = None

        self.ts = ts
        self.zdata = None
        self.scaler = scaler

    def scale_data(self):
        self.zdata = self.scaler.fit_transform(self.ts)

    def assess_stationarity(self, test=['adfuller', 'ADFuller variance'], freq=30):

        fig, ax = stt.visualize_signal_stationarity(self.ts, freq=freq)
        ax[0].scatter(self.ts.index, self.ts)
        ax[0].set_title("Visual Assessment of Signal Stationarity")

        if test is not None:
            for te in test:
                if te.lower() == "adfuller":
                    stt.adfuller_test(self.ts)
                elif te.lower() == 'levene':
                    stt.levene_test(self.ts)
                elif te.lower() == 'adfuller variance':
                    stt.adfuller_test(self.ts.rolling(30).var().dropna())
                else:
                    raise ValueError(f"Statistical test {te} not implemented")


    def default_gev_model(self):

        self.model = model_gev_linear(self.zdata)


    def sample_prior(self, samples=1000):
        self.idata = None
        self.idata = pm.sample_prior_predictive(samples=samples, model=self.model)


    def infer_posterior(self,
                        samples=2000,
                        initvals={"alpha_mu": -0.5, "beta_mu":0, "alpha_sig": 0, "beta_sig":0, "xi":0 }):
        with self.model:
            self.trace = pm.sample(
                samples,
                cores=4,
                chains=4,
                tune=2000,
                initvals=initvals,
                target_accept=0.98,
            )
        # add trace to existing idata object
        self.idata.extend(self.trace)

    def evaluate_posterior(self):
        self.idata.extend(pm.sample_posterior_predictive(self.idata, model=self.model))
        self.idata.extend(pm.compute_log_likelihood(self.idata, model=self.model))

        fig, ax = plt.subplots(1,2, figsize=(12, 3))
        az.plot_bpv(self.idata,  kind="t_stat", t_stat=lambda x:np.percentile(x, q=50, axis=-1), ax=ax[0])
        az.plot_loo_pit(idata=self.idata, y="gev", ecdf=True, ax=ax[1])

    def plot_posterior(self, var_to_plot=["alpha_mu", "beta_mu", "alpha_sig", "beta_sig", "xi", "mu", "sig"]):
        az.plot_trace(self.idata, var_names=var_to_plot,  figsize=(12, 12));
