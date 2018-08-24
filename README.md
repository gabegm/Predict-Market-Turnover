# README

## Installation
```
conda env create -f environment.yml
```

## Getting Started

### Tidy
```python
python src/tidy.py -f 'data/bundesliga.csv'
python src/tidy.py -f 'data/result.csv'
```

### Train test
```python
python src/train_test.py -f 'data/bundesliga.parquet'
```

### Train
```python
python src/train.py -f 'data/bundesliga.parquet' 'data/result.parquet'
```

### Predict
```python
python src/predict.py -f 'data/result.parquet' -m 'model/mdl.obj'
```

### Results
```python
python src/merge.py -f 'data/result.csv'
```