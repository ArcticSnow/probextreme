<!-- markdownlint-disable -->

<a href="../../probextreme/stat_test.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `probextreme.stat_test`
Collection of statistical test usefull to caracterise the data S. Filhol, December 2024 


---

<a href="../../probextreme/stat_test.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `visualize_signal_stationarity`

```python
visualize_signal_stationarity(data, freq=10)
```

Function to quickly visualize stationarity of the data using rolling mean, median and standard deviation 


---

<a href="../../probextreme/stat_test.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `adfuller_test`

```python
adfuller_test(data, significance_level=0.05)
```

Perform an Augmented Dicker-Fuller test for testing the stationarity of a signal 



**Args:**
 
 - <b>`data`</b> (array or timeseries):  data to test 
 - <b>`significance_level`</b> (float):  significance level to compare P-value. Default is 0.05 

External resource: https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test 


---

<a href="../../probextreme/stat_test.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `levene_test`

```python
levene_test(data, split_index=None, significance_level=0.05)
```

It tests the null hypothesis that the population variances are equal 



**Args:**
 
 - <b>`data`</b> (array or timeseries):  data to test 
 - <b>`split_index`</b> (int):  index to split the data in two distinct groups 
 - <b>`significance_level`</b> (float):  significance level to compare P-value. Default is 0.05 

External resource: https://en.wikipedia.org/wiki/Levene%27s_test 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
