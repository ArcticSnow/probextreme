"""
S. Filhol and F. Doussot, Sept. 2024


"""
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.stats import genextreme, gumbel_r, genpareto
import numpy as np
import multiprocessing as mp
from functools import partial
import pdb
from scipy import stats


class extreme_values:
    def __init__(self,ts,
                 BM_window='365.25D',
                 origin_BM='start',
                 POT_threshold=None,
                 mtd='96H',
                 models=['gev','gumbel','gpd'],
                 verbose=True):
        """
        Class for analyzing a timeseries using the extreme value theory. Available approaches are Block Maxima (GEV) and Threshold (GPD).

        Args:
            ts (float): pandas timeseries of values to analyze
            BM_window (freq): Block maxima window, default is 1 year
            origin_BM (str): date of the origin for the BM computation. Default 'start'
            POT_threshold (float):
            mtd (freq):
            models (str): list
        """
        #calculate Maxima (Bloc maxima or peak over threshold) :
        self.ts = ts
        self.mtd = mtd
        self.models = models
        self.model_name_dict = {'gev':'GEV','gumbel':'Gumbel', 'gpd':'GPD'}
        self.ts_BM = get_BM_values(ts, freq = BM_window, origin=origin_BM)
        self.verbose = verbose

        if 'gev' in self.models:
            self.gev = model(self.ts_BM, model_type = 'gev')
            self.gev.fit_distribution(verbose=self.verbose)

        if 'gumbel' in self.models:
            self.gumbel = model(self.ts_BM, model_type = 'gumbel')
            self.gumbel.fit_distribution(verbose=self.verbose)

        if 'gpd' in self.models:
            if POT_threshold is None:
                POT_threshold = np.percentile(ts, 75)
            self.ts_POT = get_POT_values(ts, threshold = POT_threshold , mtd = mtd)
            self.gpd.threshold = POT_threshold
            self.gpd = model(self.ts_POT, model_type = 'gpd')
            self.gpd.nb_years = ts.index.max().year - ts.index.min().year

            self.gpd.fit_distribution(verbose=self.verbose)

        self.return_periods = None
        self.return_levels = None
        self.return_periods_CI = None
        self.confidence_level = None

        # set default color scheme
        cmap = ["#3a86ff","#8338ec","#ff006e","#fb5607","#ffbe0b"]
        model_colormap = {'gev':cmap[0],
                          'gumbel':cmap[1],
                          'gpd':cmap[2]}
        model_marker_type = {'gev':"o",
                          'gumbel':"d",
                          'gpd':"x"}
        for mod in self.models:
            getattr(self, mod).color = model_colormap.get(mod)
            getattr(self, mod).marker = model_marker_type.get(mod)

    def redefine_threshold(self,threshold):
        if self.verbose:
            print(f"\n---> The threshold has been set to be {threshold:4.2f}")
        self.ts_POT = get_POT_values(self.ts, threshold=threshold, mtd=self.mtd)
        self.gpd.threshold = threshold
        self.gpd = model(self.ts_POT, model_type='gpd')
        self.gpd.fit_distribution(verbose=False)

    def find_POT_threshold(self, threshold_range = None, mtd = None, plot = 'plt', nb_peaks_min=None):
        """
        Function to find POT threshold based on the mean excess and threshold value.
        The threshold value to choose should be as small as possible in order to get some linearity in the mean residual life plot
        Args:
            threshold_range (float): numpy array of threshold for which to compute mean excess
            mtd : cf get_POT_values
        """
        verbose = self.verbose
        self.verbose = False

        if mtd is None:
            mtd = self.mtd
        if threshold_range is None :
            min_threshold = min(self.ts) * 1.5
            max_threshold = max(self.ts) * 0.9
            threshold_range = np.linspace(min_threshold, max_threshold, 100)
        if nb_peaks_min is None:
            nb_peaks_min = int(self.gpd.nb_years/4)
        #for mean residual life plot :
        mean_excess = []
        threshold_limited = []
        #for QQ-plot :
        df = pd.DataFrame(index=threshold_range, columns=['R2', 'nb_peaks'])
        print(threshold_range)

        for threshold in threshold_range :
            # Mean Residual life plot :
            ts_POT = get_POT_values(self.ts, threshold, mtd = mtd)
            excess = ts_POT - threshold
            mean_excess.append(sum(excess) / len(excess))
            threshold_limited.append(threshold)

            #Optimize QQ-plot :
            self.redefine_threshold(threshold)
            r2 = self.plot_QQ_plot(plot = None, models = ['gpd'])
            nb_peaks = self.ts_POT.shape[0]
            df['R2'].loc[threshold] = r2
            df['nb_peaks'].loc[threshold] = int(nb_peaks)
            if (nb_peaks < nb_peaks_min):
                break
        df.fillna(0)
        df['R2_float'] = df['R2'].astype('float64')
        print(df)
        opt_threshold = df['R2_float'].idxmax()

        #set the best value :
        self.verbose = verbose
        self.redefine_threshold(opt_threshold)

        if plot == 'plt' :
            plt.close()
            plt.figure()
            plt.plot(threshold_limited, mean_excess, label = 'mean residual')
            plt.title("Mean residual life plot")
            plt.axvline(x=opt_threshold, color = 'red', linestyle='--', label = "Optimal threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Mean Excess")
            plt.legend()
            plt.show()
        return opt_threshold

    def get_return_levels(self, return_periods=None):
        """
        Function to compute return levels for given return periods
        Args:
            return_periods (float): numpy array of return periods
        """
        if self.return_periods is None and return_periods is None:
            rt = np.array([2, 5, 10, 20, 50, 100])
            self.return_periods = rt
        elif return_periods is None and self.return_periods is not None:
            rt = self.return_periods
        elif return_periods is not None and self.return_periods is None:
            rt = return_periods
            self.return_periods = rt

        for mod in self.models:
            getattr(self, mod).return_levels = getattr(self, mod).get_return_levels(rt)

    def execute_bootstrapping(self, n_iterations=1000, confidence=95):
        """
        Function to compute bootstrapping method to derive confidence intervals of return levels and return periods
        Args:
            n_iterations (int): number of boostrap iteration
            confidence (float): confidence level. between 0 and 100%
        """
        if self.return_periods_CI is None:
            self.return_periods_CI = np.array([2, 5, 10, 20, 50, 100])

        self.confidence_level = confidence
        lower_percentile = (100 - confidence) / 2
        upper_percentile = 100 - lower_percentile

        print(f'---> compute return level confidence intervals at {confidence}%')
        for mod in self.models:
            getattr(self, mod).CI_levels, getattr(self, mod).ds_bootstrap = getattr(self, mod).bootstrap_return_levels(return_periods=self.return_periods_CI,
                                                                  n_bootstrap=n_iterations,
                                                                  lower_percentile=lower_percentile,
                                                                  upper_percentile=upper_percentile)
    #def compute_probability_plots(self):
    #    for mod in self.models:


    def print_summary(self):
        print()
        str_title = 'Period'
        for mod in self.models:
            str_title = str_title + "          " + self.model_name_dict.get(mod)
        print(str_title)

        for period in self.return_periods:
            list_to_print = f'{period:4.0f}'
            for mod in self.models:
                str_value = f'{getattr(self, mod).return_levels[self.return_periods==period][0]:9.2f}'
                list_to_print = list_to_print + "    " + str(str_value)
            print(list_to_print)


    def plot_return_levels(self, ax):
        #plot return periods & return levels :
        for mod in self.models:
            mod_mame = self.model_name_dict.get(mod)
            ax.plot(self.return_periods, getattr(self, mod).return_levels, label=f'{mod_mame}', color=getattr(self, mod).color)
            ax.fill_between(self.return_periods_CI, getattr(self, mod).CI_levels[0], getattr(self, mod).CI_levels[1],
                            alpha=0.1, color=getattr(self, mod).color, label=f'{self.confidence_level}% {mod_mame} confidence interval')

        ax.set_xlabel("return period (year)")
        ax.set_ylabel("return level")
        ax.legend(loc='upper left')
        return ax

    def plot_distribution(self, ax):
        h1 = None

        if ('gev' or 'gumbel') in self.models:
            mod = [item for item in ['gev','gumbel'] if item in self.models][0]
            ax.hist(self.ts_BM, bins=30, density=True, alpha=0.5, color=getattr(self, mod).color, edgecolor=None, label = 'BM values', zorder=0)
        if 'gpd' in self.models:
            ax.hist(self.ts_POT , bins=30, density=True, alpha=0.5, color=self.gpd.color, edgecolor=None, label = 'POT values', zorder=1)

        for mod in self.models:
            mod_name = self.model_name_dict.get(mod)
            x_values = np.linspace(0, np.max(self.ts_BM) * 1.2, 100)
            if mod in ['gev', 'gumbel']:
                getattr(self, mod).compute_pdf(x_values)
                y_val = getattr(self, mod).pdf

            elif mod == 'gpd':
                x_values = x_values[x_values>=self.gpd.threshold]
                getattr(self, mod).compute_pdf(x_values)
                y_val = getattr(self, mod).pdf

            ax.plot(x_values, y_val, label=mod_name, color=getattr(self, mod).color)

        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.legend(loc='upper left')
        return ax

    def plot_diagnostic(self):
        fig, ax = plt.subplots(2, 1)
        _ = self.plot_return_levels(ax[0])
        _ = self.plot_distribution(ax[1])
        plt.tight_layout()
        plt.show()

    def plot_extremes(self):
        plt.plot(self.ts.index, self.ts, color="b", label='time serie', lw=1, alpha = 0.5)
        if ('gev' or 'gumbel') in self.models:
            mod = [item for item in ['gev','gumbel'] if item in self.models][0]
            plt.scatter(self.ts_BM.index, self.ts_BM, marker="o", color=getattr(self, mod).color, label="BM values")
        if 'gpd' in self.models:
            plt.axhline(self.gpd.threshold, color='red', linestyle='dashed', label="threshold")
            plt.scatter(self.ts_POT.index, self.ts_POT, marker="x", color=self.gpd.color, label="POT values")
        plt.legend()
        plt.show()

    def plot_QQ_plot(self, plot = 'plt', models = None):
        """
        :param plt: 'plt' to plot the graph
        :return: r_value**2
        """
        if models == None :
            models = self.models
        if plot == 'plt' :
            plt.figure(figsize=(6, 6))
        for mod in models:
            if mod =='gev':
                res = stats.probplot(self.ts_BM, dist="genextreme", sparams=(self.gev.shape, self.gev.loc, self.gev.scale),
                                     plot=None, rvalue=True)
            elif mod == 'gumbel':
                res = stats.probplot(self.ts_BM, dist="gumbel_r", sparams=(self.gumbel.loc, self.gumbel.scale),
                                     plot=None, rvalue=True)
            elif mod == 'gpd':
                res = stats.probplot(self.ts_POT - self.gpd.threshold, dist="genpareto", sparams=(self.gpd.shape, self.gpd.loc, self.gpd.scale),
                                     plot=None, rvalue=True)
            theorical_quantiles = res[0][0]
            empirical_quantiles = res[0][1]
            slop, incerception, r_value = res[1]
            r2 = r_value**2
            if plot == 'plt' :
                plt.scatter(theorical_quantiles, empirical_quantiles, color=getattr(self, mod).color,
                            label=f"{mod} : R² = {r2}", marker = getattr(self, mod).marker, s=20)
                plt.plot(theorical_quantiles, slop * theorical_quantiles + incerception,
                         color=getattr(self, mod).color, label="_"+ mod, linestyle = 'dotted')
        if plot == 'plt' :
            plt.title("Q-Q plot")
            plt.xlabel('Theorical quantiles')
            plt.ylabel('Empirical quantiles')
            plt.legend()
            plt.show()
        return r2



class model:
    """
    Python class to perform GEV analysis on annual extreme values
    WARNING: currently only support maximum extremes

    TODO:
    - [ ] add plotting tools
    - [ ] print summary function
    """

    def __init__(self, ts, model_type='gev'):
        """
        Python class to perform GEV analysis on annual extreme values
        WARNING: currently only support maximum extremes

        Args:
            ts: annual extremes (maximum)
            model_type:
        """
        self.ts = ts
        self.threshold = None
        self.model_type = model_type
        self.nb_years = None
        self.pdf = None
        self.shape = None
        self.loc = None
        self.scale = None
        self.aic = None
        self.n_jobs = mp.cpu_count()
        self.levels_conf_int = None
        self.periods_conf_int = None
        self.bootstrap_samples = None

    def fit_distribution(self, verbose=True, **kwargs):
        if self.model_type.lower() == 'gev' :
            self.shape, self.loc, self.scale = genextreme.fit(self.ts, **kwargs)
            log_likelihood = np.sum(genextreme.logpdf(self.ts, self.shape, loc=self.loc, scale=self.scale))
            k = 3
            self.ks_stat, self.p_value = stats.kstest(self.ts, "gumbel_r", args=(self.loc, self.scale))
        elif self.model_type.lower() == 'gumbel':
            self.loc, self.scale = gumbel_r.fit(self.ts, **kwargs)
            self.shape = 0
            log_likelihood = np.sum(gumbel_r.logpdf(self.ts, loc=self.loc, scale=self.scale))
            k = 2
            self.ks_stat, self.p_value = stats.kstest(self.ts, "genextreme", args = (self.shape, self.loc, self.scale))
        elif self.model_type.lower() == 'gpd':
            self.shape, self.loc, self.scale = genpareto.fit(self.ts-self.threshold, floc=0, **kwargs)
            #self.shape, self.loc, self.scale = genpareto.fit(self.ts-self.threshold, **kwargs)
            log_likelihood = np.sum(genpareto.logpdf(self.ts-self.threshold, self.shape, loc=self.loc, scale=self.scale))
            k = 3
            self.ks_stat, self.p_value = stats.kstest(self.ts - self.threshold, "genpareto", args = (self.shape, self.loc, self.scale))
        else:
            raise ValueError("Distribution model not available. Choose among ['gev','gumbel','gpd']")

        self.aic = 2 * k - 2 * log_likelihood
        if verbose:
            print(f'---> {self.model_type} fitted to data')
            print(f'\tFitting parameters: shape, loc, scale = {self.shape}, {self.loc}, {self.scale}')
            print(f'\tGoodness of fit AIC: {self.aic}')
            print(f'\tKS-test : ks_stat = {self.ks_stat}, p_value = {self.p_value}')


    def compute_pdf(self, x_values=None):

        if x_values is None:
            x_values = np.linspace(0, np.max(self.ts)*1.2, 500)

        if (self.model_type.lower() == 'gev') | (self.model_type.lower() == 'gumbel'):
            self.pdf = genextreme.pdf(x_values, self.shape, self.loc, self.scale)
        elif self.model_type.lower() == 'gpd':
            self.pdf = genpareto.pdf(x_values - self.threshold, self.shape, self.loc, self.scale)
        else:
            raise ValueError("Distribution model not available. Choose among ['gev','gumbel','gpd']")
        return self.pdf

    def get_return_periods(self, return_levels=None):

        if return_levels is None:
            return_levels = np.linspace(np.min(self.ts), np.max(self.ts)*1.2, 10)

        if (self.model_type.lower() == 'gev') | (self.model_type.lower() == 'gumbel'):
            return_periods = 1 / (1 - genextreme.cdf(return_levels, self.shape , self.loc, self.scale))
        elif self.model_type.lower() == 'gpd':
            P_exceed = prob_exceedance_GPD(self.shape, self.scale, self.threshold, return_levels)
            Nu = len(self.ts)
            return_periods = self.nb_years / (Nu * P_exceed)
        else:
            raise ValueError("Distribution model not available. Choose among ['gev','gumbel','gpd']")
        return return_periods

    def get_return_levels(self, return_periods=None):
        """
        Function to get return levels of corresponding return periods based on the fitted model
        Args:
            return_periods:

        Returns:
            numpy array - return levels
        """
        if return_periods is None:
            return_periods = np.array([2, 5, 10, 20, 50, 100])
        if (self.model_type.lower() == 'gev') | (self.model_type.lower() == 'gumbel'):
            return_levels = genextreme.isf(1/return_periods, self.shape , self.loc, self.scale)
        elif self.model_type.lower() == 'gpd':
            excess_per_year = len(self.ts)/self.nb_years
            return_levels = self.threshold + (self.scale / self.shape) * ((return_periods * excess_per_year) ** self.shape - 1)
        return return_levels


    def bootstrap_return_levels(self, return_periods=None, n_bootstrap=1000, lower_percentile=2.5, upper_percentile=97.5, **fit_kwargs):
        """
        Function to compute confidence intervals (CI) of return levels at set return periods using boostrap method

        Args:

            return_periods (float): array of return periods for which to compute CI of return levels
            n_bootstrap (int): number of boostrap iterations
            upper_percentile (float):  upper percentile to extract. Default = 0.95
            lower_percentile (float): lower percentile to extract. Default = 0.05
            **fit_kwargs: kwargs for distribution to fit. See function genextreme for GEV, and gumbel_r of Scipy

        Returns:
            array of the lower and upper percentile return levels

        """
        if return_periods is None:
            return_periods = np.array([2, 5, 10, 20, 50, 100])

        self.bootstrap_samples = np.random.choice(self.ts, size=(n_bootstrap, len(self.ts)), replace=True)
        if self.model_type.lower() in ['gev', 'gumbel']:
            get_return_levels_partial = partial(
                get_return_levels, return_periods=return_periods, model=self.model_type, **fit_kwargs)
        elif self.model_type.lower() == 'gpd':
            get_return_levels_partial = partial(
                get_return_levels, return_periods=return_periods, model=self.model_type,
                threshold=self.threshold, nb_years=self.nb_years, **fit_kwargs)
        with mp.Pool(self.n_jobs) as pool:
            bootstrap_levels, shapes = zip(*pool.map(get_return_levels_partial, self.bootstrap_samples))

        res = np.array(bootstrap_levels)
        ds_bootstrap = xr.Dataset(
            data_vars=dict(
                return_levels = (["sample", "return_period"], res),
                bootstrap_samples=(["sample", "ind_data"], self.bootstrap_samples)
            ),
            coords=dict(
                sample=("sample", np.arange(0,n_bootstrap)),
                return_period=("return_period", return_periods),
                ind_data=("ind_data", np.arange(0,len(self.ts)))
            ),
            attrs=dict(
                description=f"Bootstrapping output for the model {self.model_type.lower()} using {n_bootstrap} bootstrap",
                ind_data="index of th"

            )
        )
        #pdb.set_trace()
        #delete nan values for percentile :
        res_filt=res[np.array(shapes) < 1, :]
        if (res.shape[0] - res_filt.shape[0] > 0):
            print(f"WARNING :{self.model_type} : {res.shape[0]-res_filt.shape[0]}  bootstrap results have been deleted "
              f"({(res.shape[0]-res_filt.shape[0])/n_bootstrap * 100}%) based on: scale > 1")
        self.levels_conf_int = np.percentile(res_filt, [lower_percentile, upper_percentile], axis=0)
        return self.levels_conf_int, ds_bootstrap

    def bootstrap_distribution_parameters(self, n_bootstrap=1000, **fit_kwargs):
        """
        Function to study the confidence intervals on the distribution parameters

        Args:
            n_bootstrap:
            **fit_kwargs:
        """

        bootstrap_samples = np.random.choice(self.ts, size=(n_bootstrap, len(self.ts)), replace=True)

        fit_model_partial = partial(fit_model, model=self.model_type, **fit_kwargs)

        with mp.Pool(self.n_jobs) as pool:
            bootstrap_params = pool.map(fit_model_partial, bootstrap_samples)

        res = np.array(bootstrap_params)
        c_bootstrap, loc_bootstrap, scale_bootstrap = res.T
        conf_level = 0.95
        lower_percentile = (1 - conf_level) / 2 * 100
        upper_percentile = 100 - lower_percentile

        c_conf_int = np.percentile(c_bootstrap, [lower_percentile, upper_percentile])
        loc_conf_int = np.percentile(loc_bootstrap, [lower_percentile, upper_percentile])
        scale_conf_int = np.percentile(scale_bootstrap, [lower_percentile, upper_percentile])

        gev_params = genextreme.fit(observed_data)
        c_hat, loc_hat, scale_hat = gev_params
        print(f"Estimated GEV parameters from observed data: c={c_hat}, loc={loc_hat}, scale={scale_hat}")
        print(f"95% Confidence interval for shape parameter c: {c_conf_int}")
        print(f"95% Confidence interval for location parameter loc: {loc_conf_int}")
        print(f"95% Confidence interval for scale parameter scale: {scale_conf_int}")

    def probability_exceedance_GPD(self, value):
            return prob_exceedance_GPD(self.shape, self.scale, self.threshold, value)


### Functions used for bootstraping
def fit_model(data, model='gev', **kwargs):
    """
    Function to fit GEV, Gumbel or Pareto distribution model to data. Fitting is performed with Scipy method. See scipy documentation

    Args:
        data (float): numpy array of float
        model (str): name of distribution to fit
        **kwargs: keyword for additional arguments of the fit function

    Returns:

    """
    if model == 'gev':
        shape, loc, scale = genextreme.fit(data, **kwargs)
        log_likelihood = np.sum(genextreme.logpdf(data, shape, loc, scale))
        k = 3
    elif model == 'gumbel':
        loc, scale = gumbel_r.fit(data, **kwargs)
        shape = 0
        log_likelihood = np.sum(gumbel_r.logpdf(data, loc, scale))
        k = 2
    elif model == 'gpd':
        shape, loc, scale = genpareto.fit(data, floc=0, **kwargs)
        #shape, loc, scale = genpareto.fit(data, **kwargs)
        log_likelihood = np.sum(genpareto.logpdf(data, shape, loc, scale))
        k = 3
    aic = 2 * k - 2 * log_likelihood
    #if fit does not work properly :
    return shape, loc, scale

def get_return_periods(data, return_levels, model='gev', threshold=None, nb_years=None,  **kwargs):
    if model in ['gev', 'gumbel']:
        shape, loc, scale = fit_model(data, model, **kwargs)
        return_periods = 1 / (1 - genextreme.cdf(return_levels, shape , loc, scale))
    elif model == 'gpd':
        shape, loc, scale = fit_model(data-threshold, model, **kwargs)
        return_levels[return_levels < threshold] = np.nan
        P_exceed = prob_exceedance_GPD(shape, scale, threshold, return_levels)
        Nu = len(data)
        return_periods = nb_years / (Nu * P_exceed)
    return return_periods

def get_return_levels(data, return_periods, model='gev',threshold=None, nb_years=None, **kwargs):
    if model in ['gev', 'gumbel']:
        shape, loc, scale = fit_model(data, model, **kwargs)
        return_levels = genextreme.isf(1/return_periods, shape , loc, scale)
    elif model == 'gpd':
        shape, loc, scale = fit_model(data-threshold, model, **kwargs)
        excess_per_year = len(data) / nb_years
        return_levels = threshold + (scale / shape) * ((return_periods * excess_per_year) ** shape - 1)
    return return_levels, shape

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
    def max_with_date(group):
        return group.idxmax()
    date_BM = ts.resample(freq, origin=origin).apply(max_with_date)
    ts_BM = ts[date_BM]
    return ts_BM

def prob_exceedance_GPD(shape, scale, threshold, value):
    if shape != 0 :
        a = (1 + (shape * (value - threshold)/scale))
        if a > 0 :
            prob = (1 + (shape * (value - threshold)/scale))**(-1/shape)
        else :
            prob = np.nan
    else :
        prob = np.exp(-(value - threshold))
    return prob


