CUDA_VISIBLE_DEVICES=2,3

SCARED_DIR='/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
HAMLYN_DIR='/data_hdd2/users/zhouzanwei/data/Medical/Endo/hamlyn-EDM/hamlyn'

python train_end_to_end.py --data_path $SCARED_DIR --log_dir './logs/dp' --use_dp