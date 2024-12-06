"""
Collection of statistical test usefull to caracterise the data
S. Filhol, December 2024

"""
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


def visualize_signal_stationarity(data, freq=10):
	"""
	Function to quickly visualize stationarity of the data using rolling mean, median and standard deviation
	"""

	fig, ax = plt.subplots(2,1, sharex=True)
	ax[0].plot(data.rolling(freq, center=True).median(), label='median')
	ax[0].plot(data.rolling(freq, center=True).mean(), label='mean')
	ax[1].plot(data.rolling(freq, center=True).std())

	ax[0].grid(linestyle='--', alpha=0.5)	
	ax[1].grid(linestyle='--', alpha=0.5)

	ax[0].set_ylabel(f"Rolling mean & median (window={freq})")
	ax[1].set_ylabel(f"Rolling variance (window={freq})")

	ax[0].legend()



# test stationarity of data
def adfuller_test(data, significance_level=0.05):
	"""
	Perform an Augmented Dicker-Fuller test for testing the stationarity of a signal
	
	Args:
		data (array or timeseries): data to test
		significance_level (float): significance level to compare P-value. Default is 0.05

	External resource: https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
	"""
	result = adfuller(data)

	print("=================================================================")
	print("         Augmented Dickey-Fuller test  - test stationarity    ")
	print("----------------------------------------------------------------")
	print('ADF Statistic:', result[0])
	print('p-value:', result[1])
	print('Critical Values:')
	for key, value in result[4].items():
	    print('\t%s: %.3f' % (key, value))

	print("-- Null hypothesis:  the data has a unit root and is non-stationary --")
	if p_value > significance_level:
	    print("Fail to reject the null hypothesis: => the data has a unit root and is non-stationary")
	else:
	    print("Reject the null hypothesis: => the data does not have a unit root and is stationary")


# test stationarity of variance
def levene_test(data, split_index=None, significance_level=0.05):
	"""
	It tests the null hypothesis that the population variances are equal

	Args:
		data (array or timeseries): data to test
		split_index (int): index to split the data in two distinct groups
		significance_level (float): significance level to compare P-value. Default is 0.05

	External resource: https://en.wikipedia.org/wiki/Levene%27s_test
	"""
	
	if split_index is None:
		n = len(data)
		split_index = n//2
		print("Data are by default split in two")

	data_first_half = data[:split_index]
	data_second_half = data[split_index:]

	# Perform Levene's test
	stat, p_value = stats.levene(data_first_half, data_second_half)

	print("=================================================================")
	print("         Levene’s Test  - test of variance stationarity    ")
	print("----------------------------------------------------------------")
	print('Levene’s Test Statistic:', stat)
	print('p-value:', p_value)

	print("-- Null hypothesis: Variances are equal --")
	if p_value > significance_level:
	    print("Fail to reject the null hypothesis: => Variances are equal over the two data bloc.")
	else:
	    print("Reject the null hypothesis: => Variances are changing over time.")