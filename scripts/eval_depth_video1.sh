export CUDA_VISIBLE_DEVICES=3

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/hamlyn-EDM/hamlyn'

model_type=endodav
log_dir=./logs/20250225/lorassb-dr1e-4
model_dir=$log_dir/$model_type/models/weights_2

# python evaluate_depth_video_pose.py --model_type $model_type \
#     --data_path $SCARED_DIR --eval_split scared_video \
#     --load_weights_folder $model_dir --eval_mono --visualize_depth \
#     --disable_residual_block --disable_conv_head --lora_type=ssb

python evaluate_depth_video_hamlyn.py --model_type $model_type \
    --data_path $HAMLYN_DIR --eval_split hamlyn_video \
    --load_weights_folder $model_dir --eval_mono --visualize_depth \
    --disable_residual_block --disable_conv_head --lora_type=ssb