# Extreme Probability Modeling


The module `from probextreme import frequentist_extreme as fe` allows to analyse a dataset under the scope of Extreme Value Theory using a frequentist approach. 


## Basic Usage Example (freqentist approach)

```python
import pandas as pd
from probextreme import frequentist_extreme as fe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import you data. Here we assume the data contain a column called 'rain'. Index must be datetime.
df = pd.read_csv('mydata.csv')


# Initialize with the time series df.rain
ev_rain = fe.extreme_values(df.rain, BM_window='365.25D', POT_threshold=45, mtd='48H')

ev_rain.find_POT_threshold()
ev_rain.plot_extremes()
ev_rain.get_return_levels(return_periods=np.arange(2, 101, 2))
ev_rain.return_periods_CI = np.arange(2, 101, 2)

# Bootstrapping to evaluate uncertainty of fit
ev_rain.execute_bootstrapping(n_iterations=1000)
ev_rain.print_summary()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ev_rain.plot_return_levels(ax[0])
ev_rain.plot_distribution(ax[1])

```

### Things to be aware

1. The argument `BM_window` is a Pandas freq type of variable. If using `365.25D`, it will use the first timestamp of the timeseries as origin for the Block Maxima computation, whereas if using `1Y`, it will use a calendar year as reference.
2. If the distribution 'GEV' does not converge during the fit, try using the argument `loc` equal the `loc` from the Gumbel distribution fit. Loc will then be used as initial value for the fit and is more likely to converge on a credible fit.


## Resources

Coles, S., Bawa, J., Trenner, L., & Dorazio, P. (2001). An introduction to statistical modeling of extreme values (Vol. 208, p. 208). London: Springer. doi: [10.1007/978-1-4471-3675-0](https://link.springer.com/book/10.1007/978-1-4471-3675-0)



