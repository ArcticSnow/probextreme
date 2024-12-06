<!-- markdownlint-disable -->

<a href="../../probextreme/bayesian_extreme.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `probextreme.bayesian_extreme`
Functionalities using Bayesian approch for extreme value analysis 


---

<a href="../../probextreme/bayesian_extreme.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bayesian_stationary_gev`

```python
bayesian_stationary_gev(
    ts,
    return_periods=array([  2,   5,  10,  20,  50, 100]),
    return_levels=11
)
```

Function to fit GEV using the Bayesian approach. This model assume data stationarity 



**Args:**
 
 - <b>`ts`</b> (timeseries, array):  data to be fitted by stationary GEV 
 - <b>`return_periods`</b> (int array):  return period to compute return level for 
 - <b>`return_levels`</b> (int, array):  return level to compute return period for 

Return: model, idata, scaler 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
