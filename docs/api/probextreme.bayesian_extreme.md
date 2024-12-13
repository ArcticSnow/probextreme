<!-- markdownlint-disable -->

<a href="../../probextreme/bayesian_extreme.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `probextreme.bayesian_extreme`
Functionalities using Bayesian approch for extreme value analysis Simon Filhol, December 2024 


---

<a href="../../probextreme/bayesian_extreme.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bayesian_stationary_gev`

```python
bayesian_stationary_gev(
    ts,
    return_periods=array([  2,   5,  10,  20,  50, 100]),
    return_levels=11
)
```

Function to fit GEV using the Bayesian approach. This model assumes data stationarity 



**Args:**
 
 - <b>`ts`</b> (timeseries, array):  data to be fitted by stationary GEV 
 - <b>`return_periods`</b> (int array):  return period to compute return level for 
 - <b>`return_levels`</b> (int, array):  return level to compute return period for 

Return: model, idata, scaler 


---

<a href="../../probextreme/bayesian_extreme.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `model_gpd_linear`

```python
model_gpd_linear(zdata)
```






---

<a href="../../probextreme/bayesian_extreme.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `model_gev_linear`

```python
model_gev_linear(zdata)
```

A Bayesian GEV model with Loc and Shape being linearly time dependent 



**Args:**
 
 - <b>`zdata`</b> (array):  standardized data to use for Bayesian inference 



**Returns:**
 model (pymc model) 


---

<a href="../../probextreme/bayesian_extreme.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Bayesian_Extreme`
Class to perform bayesian modeling of extreme values with by default time dependence. This class is wrapper of PyMC method and intends to provide a basic model. For advanced controls on the parameters, refer to PyMC methods on top of which this class is built. 



**Attributes:**
 
 - <b>`ts`</b> (timeseries or dataframe):  time series of maximum. Default model is GEV so ts must contain block maximum values 
 - <b>`scaler`</b> (obj):  scaling object. See utils.py 

Methods: scale_data() assess_stationarity(test=['adfuller', 'ADFuller variance'], freq=30) default_gev_model() sample_prior(samples=1000) infer_posterior(samples=2000) evaluate_posterior() 

<a href="../../probextreme/bayesian_extreme.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(ts, scaler=<probextreme.utils.StandardScaler object at 0x7014ab6d4040>)
```








---

<a href="../../probextreme/bayesian_extreme.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assess_stationarity`

```python
assess_stationarity(test=['adfuller', 'ADFuller variance'], freq=30)
```





---

<a href="../../probextreme/bayesian_extreme.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `default_gev_model`

```python
default_gev_model()
```





---

<a href="../../probextreme/bayesian_extreme.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate_posterior`

```python
evaluate_posterior()
```





---

<a href="../../probextreme/bayesian_extreme.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `infer_posterior`

```python
infer_posterior(
    samples=2000,
    initvals={'alpha_mu': -0.5, 'beta_mu': 0, 'alpha_sig': 0, 'beta_sig': 0, 'ξ': 0}
)
```





---

<a href="../../probextreme/bayesian_extreme.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `plot_posterior`

```python
plot_posterior(
    var_to_plot=['alpha_mu', 'beta_mu', 'alpha_sig', 'beta_sig', 'ξ', 'μ', 'σ']
)
```





---

<a href="../../probextreme/bayesian_extreme.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sample_prior`

```python
sample_prior(samples=1000)
```





---

<a href="../../probextreme/bayesian_extreme.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `scale_data`

```python
scale_data()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
