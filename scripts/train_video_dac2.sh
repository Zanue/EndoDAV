export CUDA_VISIBLE_DEVICES=1

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'

model_type=endodac
log_dir=./logs/20250219/nores-randomtrain
model_dir=$log_dir/$model_type/models/weights_9

# python train_end_to_end_video.py --data_path $SCARED_DIR \
#     --model_type $model_type --num_workers 4 \
#     --log_dir $log_dir \
#     --batch_size 16 --T 1 --encoder vits --pre_norm \
#     --disable_residual_block

python evaluate_depth_video_pose.py --model_type $model_type \
    --data_path $SCARED_DIR --eval_split scared_video \
    --load_weights_folder $model_dir --eval_mono --visualize_depth --pre_norm \
    --disable_residual_block