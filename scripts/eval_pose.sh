export CUDA_VISIBLE_DEVICES=0 

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'
EndoDAC_model_dir=logs/video/base/endodav/endodac/models/weights_last
EndoDAV_model_dir=logs/video/new_split/endodav/models/weights_last

# python evaluate_pose.py --data_path $SCARED_DIR \
#     --load_weights_folder $EndoDAC_model_dir --eval_mono

python evaluate_pose.py --data_path $SCARED_DIR \
    --load_weights_folder $EndoDAV_model_dir --eval_mono