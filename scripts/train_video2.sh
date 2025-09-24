export CUDA_VISIBLE_DEVICES=3

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'

model_type=endodav
log_dir=./logs/20250220/scales0-reproj3-flow3
model_dir=$log_dir/$model_type/models/weights_last

python train_end_to_end_video.py --data_path $SCARED_DIR \
    --model_type $model_type --num_workers 4 \
    --log_dir $log_dir \
    --batch_size 1 --T 16 --encoder vits \
    --disable_residual_block --disable_conv_head \
    --scales 0 \
    --depth_reproj 1e-3 --depth_flow 1e-3

python evaluate_depth_video.py --model_type $model_type \
    --data_path $SCARED_DIR --eval_split scared_video \
    --load_weights_folder $model_dir --eval_mono --visualize_depth \
    --disable_residual_block --disable_conv_head