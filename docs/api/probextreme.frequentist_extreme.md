<!-- markdownlint-disable -->

<a href="../../probextreme/frequentist_extreme.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `probextreme.frequentist_extreme`
S. Filhol and F. Doussot, Sept. 2024 


---

<a href="../../probextreme/frequentist_extreme.py#L506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fit_model`

```python
fit_model(data, model='gev', **kwargs)
```

Function to fit GEV, Gumbel or Pareto distribution model to data. Fitting is performed with Scipy method. See scipy documentation 



**Args:**
 
 - <b>`data`</b> (float):  numpy array of float 
 - <b>`model`</b> (str):  name of distribution to fit 
 - <b>`**kwargs`</b>:  keyword for additional arguments of the fit function 



**Returns:**
 


---

<a href="../../probextreme/frequentist_extreme.py#L536"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_return_periods`

```python
get_return_periods(
    data,
    return_levels,
    model='gev',
    threshold=None,
    nb_years=None,
    **kwargs
)
```






---

<a href="../../probextreme/frequentist_extreme.py#L548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_return_levels`

```python
get_return_levels(
    data,
    return_periods,
    model='gev',
    threshold=None,
    nb_years=None,
    **kwargs
)
```






---

<a href="../../probextreme/frequentist_extreme.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_POT_values`

```python
get_POT_values(ts, threshold, mtd='5D')
```

Function to extract values from timseries for the GPD analysis 



**Args:**
 
 - <b>`ts`</b> (float):  time series 
 - <b>`threshold`</b> (float):  threshold over which values of ts are considered extreme values and considered in GPD processing 
 - <b>`mtd`</b> (freq):  minimal time distance between 2 max Return: POT values 


---

<a href="../../probextreme/frequentist_extreme.py#L576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_BM_values`

```python
get_BM_values(ts, freq='365.25D', origin='start')
```






---

<a href="../../probextreme/frequentist_extreme.py#L583"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `prob_exceedance_GPD`

```python
prob_exceedance_GPD(shape, scale, threshold, value)
```






---

<a href="../../probextreme/frequentist_extreme.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `extreme_values`




<a href="../../probextreme/frequentist_extreme.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    ts,
    BM_window='365.25D',
    origin_BM='start',
    POT_threshold=None,
    mtd='96H',
    models=['gev', 'gumbel', 'gpd'],
    verbose=True
)
```

Class for analyzing a timeseries using the extreme value theory. Available approaches are Block Maxima (GEV) and Threshold (GPD). 



**Args:**
 
 - <b>`ts`</b> (float):  pandas timeseries of values to analyze 
 - <b>`BM_window`</b> (freq):  Block maxima window, default is 1 year 
 - <b>`origin_BM`</b> (str):  date of the origin for the BM computation. Default 'start' POT_threshold (float): mtd (freq): 
 - <b>`models`</b> (str):  list 




---

<a href="../../probextreme/frequentist_extreme.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `execute_bootstrapping`

```python
execute_bootstrapping(n_iterations=1000, confidence=95)
```

Function to compute bootstrapping method to derive confidence intervals of return levels and return periods 

**Args:**
 
 - <b>`n_iterations`</b> (int):  number of boostrap iteration 
 - <b>`confidence`</b> (float):  confidence level. between 0 and 100% 

---

<a href="../../probextreme/frequentist_extreme.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `find_POT_threshold`

```python
find_POT_threshold(
    threshold_range=None,
    mtd=None,
    plot='plt',
    nb_peaks_min=None
)
```

Function to find POT threshold based on the mean excess and threshold value. The threshold value to choose should be as small as possible in order to get some linearity in the mean residual life plot 

**Args:**
 
 - <b>`threshold_range`</b> (float):  numpy array of threshold for which to compute mean excess 
 - <b>`mtd `</b>:  cf get_POT_values 

---

<a href="../../probextreme/frequentist_extreme.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_return_levels`

```python
get_return_levels(return_periods=None)
```

Function to compute return levels for given return periods 

**Args:**
 
 - <b>`return_periods`</b> (float):  numpy array of return periods 

---

<a href="../../probextreme/frequentist_extreme.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_QQ_plot`

```python
plot_QQ_plot(plot='plt', models=None)
```

:param plt: 'plt' to plot the graph :return: r_value**2 

---

<a href="../../probextreme/frequentist_extreme.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_diagnostic`

```python
plot_diagnostic()
```





---

<a href="../../probextreme/frequentist_extreme.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_distribution`

```python
plot_distribution(ax)
```





---

<a href="../../probextreme/frequentist_extreme.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_extremes`

```python
plot_extremes()
```





---

<a href="../../probextreme/frequentist_extreme.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_return_levels`

```python
plot_return_levels(ax)
```





---

<a href="../../probextreme/frequentist_extreme.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `print_summary`

```python
print_summary()
```





---

<a href="../../probextreme/frequentist_extreme.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `redefine_threshold`

```python
redefine_threshold(threshold)
```






---

<a href="../../probextreme/frequentist_extreme.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `model`
Python class to perform GEV analysis on annual extreme values WARNING: currently only support maximum extremes 



**TODO:**
 
- [ ] add plotting tools 
- [ ] print summary function 

<a href="../../probextreme/frequentist_extreme.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(ts, model_type='gev')
```

Python class to perform GEV analysis on annual extreme values WARNING: currently only support maximum extremes 



**Args:**
 
 - <b>`ts`</b>:  annual extremes (maximum) model_type: 




---

<a href="../../probextreme/frequentist_extreme.py#L468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bootstrap_distribution_parameters`

```python
bootstrap_distribution_parameters(n_bootstrap=1000, **fit_kwargs)
```

Function to study the confidence intervals on the distribution parameters 



**Args:**
  n_bootstrap:  **fit_kwargs: 

---

<a href="../../probextreme/frequentist_extreme.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bootstrap_return_levels`

```python
bootstrap_return_levels(
    return_periods=None,
    n_bootstrap=1000,
    lower_percentile=2.5,
    upper_percentile=97.5,
    **fit_kwargs
)
```

Function to compute confidence intervals (CI) of return levels at set return periods using boostrap method 



**Args:**
 


 - <b>`return_periods`</b> (float):  array of return periods for which to compute CI of return levels 
 - <b>`n_bootstrap`</b> (int):  number of boostrap iterations 
 - <b>`upper_percentile`</b> (float):   upper percentile to extract. Default = 0.95 
 - <b>`lower_percentile`</b> (float):  lower percentile to extract. Default = 0.05 
 - <b>`**fit_kwargs`</b>:  kwargs for distribution to fit. See function genextreme for GEV, and gumbel_r of Scipy 



**Returns:**
 array of the lower and upper percentile return levels 

---

<a href="../../probextreme/frequentist_extreme.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute_pdf`

```python
compute_pdf(x_values=None)
```





---

<a href="../../probextreme/frequentist_extreme.py#L336"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_distribution`

```python
fit_distribution(verbose=True, **kwargs)
```





---

<a href="../../probextreme/frequentist_extreme.py#L393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_return_levels`

```python
get_return_levels(return_periods=None)
```

Function to get return levels of corresponding return periods based on the fitted model 

**Args:**
  return_periods: 



**Returns:**
  numpy array - return levels 

---

<a href="../../probextreme/frequentist_extreme.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_return_periods`

```python
get_return_periods(return_levels=None)
```





---

<a href="../../probextreme/frequentist_extreme.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `probability_exceedance_GPD`

```python
probability_exceedance_GPD(value)
```






---

<a href="../../probextreme/frequentist_extreme.py#L595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StandardScaler`




<a href="../../probextreme/frequentist_extreme.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Class to define a z-score scaler with transform and inverse transform functionalities 

zdata = (data - mean(data))/std(data) 




---

<a href="../../probextreme/frequentist_extreme.py#L606"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(data)
```





---

<a href="../../probextreme/frequentist_extreme.py#L616"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(data)
```





---

<a href="../../probextreme/frequentist_extreme.py#L613"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_transform`

```python
inv_transform(zdata)
```





---

<a href="../../probextreme/frequentist_extreme.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(data)
```






---

<a href="../../probextreme/frequentist_extreme.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RobustScaler`




<a href="../../probextreme/frequentist_extreme.py#L621"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Class to define a robust z-score scaler with transform and inverse transform functionalities 

zdata = (data - median(data))/std(data) 




---

<a href="../../probextreme/frequentist_extreme.py#L631"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(data)
```





---

<a href="../../probextreme/frequentist_extreme.py#L641"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(data)
```





---

<a href="../../probextreme/frequentist_extreme.py#L638"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_transform`

```python
inv_transform(zdata)
```





---

<a href="../../probextreme/frequentist_extreme.py#L635"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(data)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
