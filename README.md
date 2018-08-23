python tidy.py -f 'data/bundesliga.csv'
python tidy.py -f 'data/result.csv'

python mdl_train_test.py -f 'data/bundesliga.parquet'

python mdl_train.py -f 'data/bundesliga.parquet' 'data/result.parquet'

python mdl_predict.py -f 'data/result.parquet' -m 'model/mdl.obj'