export CUDA_VISIBLE_DEVICES=0 

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'
model_dir=./pretrained_model/full_model

python evaluate_depth.py --data_path $SCARED_DIR --eval_split endovis \
    --load_weights_folder $model_dir --eval_mono --visualize_depth