export CUDA_VISIBLE_DEVICES=1

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/hamlyn-EDM/hamlyn'

model_type=endodav
log_dir=./logs/20250223/scales4-d1000
model_dir=$log_dir/$model_type/models/weights_6

python evaluate_depth_video_hamlyn.py --model_type $model_type \
    --data_path $HAMLYN_DIR --eval_split hamlyn_video \
    --load_weights_folder $model_dir --eval_mono --visualize_depth \
    --disable_residual_block --disable_conv_head
    # --max_length 1000