export TRAINING_DATA=input/train_folds.csv
export FOLD=0
export MODEL=$1  # To declare an argument (a déclarer quand on lance le run.sh)
#export TEST_DATA=input/test.csv



python -m src.train
#FOLD=0 python -m src.train
#FOLD=1 python -m src.train
#FOLD=2 python -m src.train
#FOLD=3 python -m src.train
#FOLD=4 python -m src.train
#python -m src.predict