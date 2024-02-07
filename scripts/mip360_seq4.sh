DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq4_inplace --reg_lambda=2 --iterations 20000
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq4_inplace --reg_lambda=2 --iterations 20000