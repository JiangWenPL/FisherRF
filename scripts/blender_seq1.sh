DATASET_PATH=$1
EXP_PATH=$2
METHOD=$3

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=${METHOD} --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=${METHOD} --seed=0 --schema v20seq1_inplace --iterations 30000  --white_background