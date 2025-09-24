export CUDA_VISIBLE_DEVICES=1

ENDONeRF_DIR=/data_hdd2/users/zhouzanwei/data/Medical/endonerf
scene=cutting_tissues_twice

depth_DA_dir=$ENDONeRF_DIR/$scene/depth_dam
depth_DA2_dir=$ENDONeRF_DIR/$scene/depth_damv2
depth_VDA_dir=$ENDONeRF_DIR/$scene/depth_vdam

python point_cloud_visualizer.py --max_frames 100 \
    --data_path $ENDONeRF_DIR/$scene \
    --depth_path $depth_VDA_dir