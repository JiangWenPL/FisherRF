DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq4_inplace --iterations 30000 --filter_out_grad rotation --densify_until_iter=10000 --densify_from_iter=500 --white_background
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=H_reg --seed=0 --schema v20seq4_inplace --iterations 30000 --filter_out_grad rotation --densify_until_iter=10000 --densify_from_iter=500 --white_background