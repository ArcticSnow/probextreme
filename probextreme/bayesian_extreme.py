"""
Functionalities using Bayesian approch for extreme value analysis

"""

import numpy as np
import arviz as az
import pymc as pm
import pymc_experimental.distributions as pmx
import pytensor.tensor as pt
from . import utils


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

        μ = pm.Normal("μ", mu=0, sigma=0.5)
        σ = pm.HalfNormal("σ", sigma=0.3)
        ξ = pm.TruncatedNormal("ξ", mu=0, sigma=0.2, lower=-0.6, upper=0.6)

        # Estimation
        gev = pmx.GenExtreme("gev", mu=μ, sigma=σ, xi=ξ, observed=zdata)
        # Return levels and return period
        z_p = pm.Deterministic("z_p",
                               μ - σ / ξ * (1 - (-pm.math.log(1 - p)) ** (-ξ)),
                               dims='probability')
        period = pm.Deterministic("return_period",
                                  1/(1-pm.math.exp(pmx.GenExtreme.logcdf(ret, mu=μ, sigma=σ, xi=ξ))),
                                  dims='return_level')


    idata = pm.sample_prior_predictive(samples=1000, model=model)
    with model:
        trace = pm.sample(
            5000,
            cores=4,
            chains=4,
            tune=2000,
            initvals={"μ": -0.5, "σ": 1.0, "ξ": -0.1},
            target_accept=0.98,
        )
    # add trace to existing idata object
    idata.extend(trace)
    idata.extend(pm.sample_posterior_predictive(trace, model=model))

    return model, idata, scaler


