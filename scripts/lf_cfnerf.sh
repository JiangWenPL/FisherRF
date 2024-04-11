DATASET_PATH=$1
EXP_PATH=$2
OBJ=$3

echo python active_train.py -s $DATASET_PATH/${OBJ} -m ${EXP_PATH} --override_idxs ${OBJ} --eval --schema=all --eval --resolution 2 --iterations 3000 --test_iterations 1000 2000 3000 --densify_until_iter=2800 --sh_up_every=1000 --sh_degree=2
python active_train.py -s $DATASET_PATH/${OBJ} -m ${EXP_PATH} --override_idxs ${OBJ} --eval --schema=all --eval --resolution 2 --iterations 3000 --test_iterations 1000 2000 3000 --densify_until_iter=2800 --sh_up_every=1000 --sh_degree=2