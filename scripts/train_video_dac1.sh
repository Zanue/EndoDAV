export CUDA_VISIBLE_DEVICES=2

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'

model_type=endodac
log_dir=./logs/20250224/disable_conv_head-dac
model_dir=$log_dir/$model_type/models/weights_last

python train_end_to_end_video.py --data_path $SCARED_DIR \
    --model_type $model_type --num_workers 8 \
    --log_dir $log_dir \
    --disable_conv_head \
    --batch_size 16 --T 1 --encoder vits --visualize_depth

python evaluate_depth_video_pose.py --model_type $model_type \
    --data_path $SCARED_DIR --eval_split scared_video \
    --load_weights_folder $model_dir --eval_mono --visualize_depth