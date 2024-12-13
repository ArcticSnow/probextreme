<!-- markdownlint-disable -->

<a href="../../probextreme/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `probextreme.utils`
Useful general functionalities S. Filhol, December 2024 


---

<a href="../../probextreme/utils.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../probextreme/utils.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_BM_values`

```python
get_BM_values(ts, freq='365.25D', origin='start')
```

 Function to extract Block Maxima values of a timeseries 



**Args:**
 
 - <b>`ts`</b> (float):  time series freq (freq):  
 - <b>`mtd`</b> (freq):  minimal time distance between 2 max Return: BM_values 


---

<a href="../../probextreme/utils.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StandardScaler`




<a href="../../probextreme/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Class to define a z-score scaler with transform and inverse transform functionalities 

zdata = (data - mean(data))/std(data) 




---

<a href="../../probextreme/utils.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(data)
```





---

<a href="../../probextreme/utils.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(data)
```





---

<a href="../../probextreme/utils.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_transform`

```python
inv_transform(zdata)
```





---

<a href="../../probextreme/utils.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(data)
```






---

<a href="../../probextreme/utils.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RobustScaler`




<a href="../../probextreme/utils.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Class to define a robust z-score scaler with transform and inverse transform functionalities 

zdata = (data - median(data))/std(data) 




---

<a href="../../probextreme/utils.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(data)
```





---

<a href="../../probextreme/utils.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(data)
```





---

<a href="../../probextreme/utils.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inv_transform`

```python
inv_transform(zdata)
```





---

<a href="../../probextreme/utils.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(data)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
