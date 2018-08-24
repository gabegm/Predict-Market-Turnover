# README

## Installation
```
conda env create -f environment.yml
```

## Getting Started

### Tidy
```python
python tidy.py -f 'data/bundesliga.csv'
python tidy.py -f 'data/result.csv'
```

### Train test
```python
python mdl_train_test.py -f 'data/bundesliga.parquet'
```

### Train
```python
python mdl_train.py -f 'data/bundesliga.parquet' 'data/result.parquet'
```

### Predict
```python
python mdl_predict.py -f 'data/result.parquet' -m 'model/mdl.obj'
```
