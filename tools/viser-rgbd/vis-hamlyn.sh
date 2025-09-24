export CUDA_VISIBLE_DEVICES=1

depth_DA_dir=/data_hdd2/users/zhouzanwei/projects/Depth/Depth-Anything-V2/outputs/endo/scared/dataset1/keyframe1/left/depth
depth_VDA_dir=/data_hdd2/users/zhouzanwei/projects/Depth/Video-Depth-Anything/outputs/endo/scared/dataset1/keyframe1/left/depth
depth_EndoDAC_dir=/data_hdd2/users/zhouzanwei/projects/Medical/EndoDAV/logs/20250219/nores-randomtrain/endodac/models/weights_last/eval/scared_video/dataset5/keyframe1/depth


python point_cloud_visualizer.py --max_frames 100 \
    --data_type hamlyn \
    --data_path /data_hdd2/users/zhouzanwei/data/Medical/hamlyn-EDM/hamlyn/rectified01 