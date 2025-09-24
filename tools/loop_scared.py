import os
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
cv2.ocl.setUseOpenCL(False)

MIN_DEPTH = 1e-3
MAX_DEPTH = 150
def read_depths(depth_dir):
    depths = []
    dep_files = sorted(os.listdir(depth_dir))
    for file in tqdm(dep_files, desc="Reading depths..."):
        depth_path = os.path.join(depth_dir, file)
        if file.endswith('.tiff'):
            depth = cv2.imread(depth_path, 3).astype(np.float32)[0:1024, :, 0]
        elif file.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        depths.append(depth)
    return np.stack(depths, axis=0) # [B, H, W]

data_root = '/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared'
for split in ['train', 'test']:
    split_dir = os.path.join(data_root, split)
    scans = sorted(os.listdir(split_dir))
    for scan in scans:
        scan_root = os.path.join(split_dir, scan)
        frames = sorted(os.listdir(scan_root))
        for frame in frames:
            frame_dir = os.path.join(scan_root, frame)
            depth_dir = os.path.join(frame_dir, 'data', 'scene_points')
            if not os.path.exists(depth_dir):
                continue
            depths = read_depths(depth_dir)
            valid_mask = np.logical_and(depths > MIN_DEPTH, depths < MAX_DEPTH)
            print(f'Seq {scan}, kf {frame}, valid depth mean: {depths[valid_mask].mean()}, valid depth std: {depths[valid_mask].std()}')
