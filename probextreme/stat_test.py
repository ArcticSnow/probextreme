"""
Collection of statistical test usefull to caracterise the data
S. Filhol

"""
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


# test stationarity of data

def adfuller_test(data)
	result = adfuller(data)


	print("=================================================================")
	print("         Levene’s Test  - test of variance stationarity    ")
	print("----------------------------------------------------------------")
	print('Levene’s Test Statistic:', stat)
	print('p-value:', p_value)
	print('ADF Statistic:', result[0])
	print('p-value:', result[1])
	print('Critical Values:')
	for key, value in result[4].items():
	    print('\t%s: %.3f' % (key, value))


# test stationarity of variance

def levene_test(data, split_index=None):
	"""
	It tests the null hypothesis that the population variances are equal

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
	if p_value > 0.05:
	    print("Fail to reject the null hypothesis: => Variances are equal over time.")
	else:
	    print("Reject the null hypothesis: => Variances are changing over time.")