export CUDA_VISIBLE_DEVICES=1

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/hamlyn-EDM/hamlyn'
VDA_Depth_Dir='/data_hdd2/users/zhouzanwei/projects/Depth/Video-Depth-Anything/outputs/endo/hamlyn'
EndoDAV_Depth_Dir='logs/20250223/scales4-d1000/endodav/models/weights_6/eval/scared_video'

python evaluate_depth_video_hamlyn.py \
    --data_path $SCARED_DIR/train --eval_split scared_video \
    --eval_mono \
    --pred_root $EndoDAV_Depth_Dir