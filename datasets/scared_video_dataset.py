from __future__ import absolute_import, division, print_function

import os
import random
import json
import glob
import numpy as np
from tqdm import tqdm
import cv2
import imageio
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_images(image_dir, max_res=-1):
    images = []
    img_files = sorted(os.listdir(image_dir), key=lambda x: int(x[:-4]))
    for file in tqdm(img_files, desc="Reading images..."):
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.JPG'):
            img_path = os.path.join(image_dir, file)
            image = imageio.imread(img_path)
            if max_res > 0 and max(image.shape[0], image.shape[1]) > max_res:
                scale = max_res / max(image.shape[0], image.shape[1])
                height = ensure_even(round(image.shape[0] * scale))
                width = ensure_even(round(image.shape[1] * scale))
                image = cv2.resize(image, (width, height))
            images.append(image)
    return np.stack(images, axis=0) # [B, H, W, C]

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

def read_poses(pose_dir):
    poses = []
    pose_files = sorted(os.listdir(pose_dir))
    for file in tqdm(pose_files, desc="Reading poses..."):
        pose_path = os.path.join(pose_dir, file)
        with open(pose_path, 'r') as path:
            data = json.load(path)
            pose = np.array(data['camera-pose']) # w2c
        poses.append(pose)
    return np.stack(poses, axis=0) # [B, 4, 4]

def load_sequence(data_path, filename):
    keyframe_dir = os.path.join(data_path, filename)
    colors = read_images(os.path.join(keyframe_dir, "data", "left"))
    depths = read_depths(os.path.join(keyframe_dir, "data", "scene_points"))
    poses = read_poses(os.path.join(keyframe_dir, "data", "frame_data"))
    assert len(colors) == len(depths) == len(poses)
    return colors, depths, poses

class SCAREDVideos(data.Dataset):
    def __init__(self, data_path, filenames, pred_root=None):
        super(SCAREDVideos, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.pred_root = pred_root
        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.pred_root is None:
            colors, depths, poses = load_sequence(self.data_path, filename)
            K = self.K.copy()
            height, width = colors.shape[1], colors.shape[2]
            K[0, :] *= width
            K[1, :] *= height
            Ks = np.stack([K] * len(colors), axis=0) # [B, 4, 4]
            return {
                "colors": colors,
                "depths": depths,
                "poses": poses,
                "Ks": Ks,
                "filename": filename
            }
        else:
            keyframe_dir = os.path.join(self.data_path, filename)
            depths = read_depths(os.path.join(keyframe_dir, "data", "scene_points"))
            pred_dir = os.path.join(self.pred_root, filename)
            pred_depths = read_depths(os.path.join(pred_dir, "depth"))
            assert len(depths) == len(pred_depths)
            return {
                "depths": depths,
                "pred_depths": pred_depths,
                "filename": filename
            }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class SCAREDRAWVideoDataset(data.Dataset):
    """Superclass for Video dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 T=-1,
                 frame_max_interval=1):
        super(SCAREDRAWVideoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        # self.interp = Image.LANCZOS
        self.interp = InterpolationMode.BILINEAR

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext
        self.T = T
        self.frame_max_interval = frame_max_interval
        self.random_train = False

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x: x

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp,
                                               antialias=True)
        self.load_depth = not self.is_train

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"l": "left", "r": "right"}

        self.data_paths_dict = self.get_data_paths(data_path, filenames)
        print(f"Loaded {len(self.data_paths_dict['images_left'])} images")

    def preprocess(self, colors, color_aug_func):
        """Resize colour images to the required scales and augment if required

        We create the color_aug_func object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        scaled_colors = [colors]
        scaled_colors_aug = []
        for i in range(self.num_scales):
            color = self.resize[i](scaled_colors[-1])
            color_aug = color_aug_func(color)
            scaled_colors.append(self.normalize(color))
            scaled_colors_aug.append(self.normalize(color_aug))
        return scaled_colors[1:], scaled_colors_aug
                
    def __len__(self):
        length = len(self.data_paths_dict['images_left']) - self.T - (len(self.frame_idxs)-1) + 1 - self.frame_max_interval*self.T
        length //= self.T
        return length

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        index = index * self.T + random.randint(0, self.T-1)
        if self.frame_max_interval > 1:
            frame_steps = np.random.randint(1, self.frame_max_interval, size=(self.T+2))
        else:
            frame_steps = np.ones(self.T+2).astype(np.int32)
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        # do_color_aug = False
        # do_flip = False
        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        else:
            color_aug = (lambda x: x)
        side = 'l'
        assert(self.frame_idxs == [0, -1, 1])
        

        inputs = {}
        if self.random_train:
            total_img_num = len(self.data_paths_dict['images_left'])
            indices = np.random.randint(self.frame_max_interval, total_img_num-self.frame_max_interval-1, size=(self.T))
            colors = self.get_color(indices, side, do_flip)
            scaled_colors, scaled_colors_aug = self.preprocess(colors, color_aug)
            for s in range(self.num_scales):
                inputs[("color", 0, s)] = scaled_colors[s]
                inputs[("color_aug", 0, s)] = scaled_colors_aug[s]
            indices_forward = indices + frame_steps[:self.T]
            colors = self.get_color(indices_forward, side, do_flip)
            scaled_colors, scaled_colors_aug = self.preprocess(colors, color_aug)
            for s in range(self.num_scales):
                inputs[("color", 1, s)] = scaled_colors[s]
                inputs[("color_aug", 1, s)] = scaled_colors_aug[s]
            indices_backward = indices - frame_steps[:self.T]
            colors = self.get_color(indices_backward, side, do_flip)
            scaled_colors, scaled_colors_aug = self.preprocess(colors, color_aug)
            for s in range(self.num_scales):
                inputs[("color", -1, s)] = scaled_colors[s]
                inputs[("color_aug", -1, s)] = scaled_colors_aug[s]
        else:
            # indices = list(range(index+1, index+self.T+1))
            # indices_all = list(range(index, index+self.T+2))
            indices_all = [index + fi*frame_steps[fi] for fi in range(self.T+2)]
            indices = indices_all[1:-1]
            colors = self.get_color(indices_all, side, do_flip)
            scaled_colors, scaled_colors_aug = self.preprocess(colors, color_aug)
            for i in self.frame_idxs:
                for s in range(self.num_scales):
                    inputs[("color", i, s)] = scaled_colors[s][1+i:self.T+1+i].clone()
                    inputs[("color_aug", i, s)] = scaled_colors_aug[s][1+i:self.T+1+i].clone()

        if self.load_depth:
            inputs["depth_gt"] = self.get_depth(indices, do_flip)
        # inputs["pose_gt"] = self.get_pose(indices, do_flip)
            
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)[None].repeat(self.T, 1, 1)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)[None].repeat(self.T, 1, 1)
            
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)[None].repeat(self.T, 1, 1)
        
        return inputs
    
    def get_data_paths(self, data_path, filenames):
        data_paths_dict = {
            'images_left': [],
            'images_right': [],
            'depths': [],
            'poses': []
        }
        for filename in filenames:
            keyframe_dir = os.path.join(data_path, filename)
            image_left_paths = sorted(glob.glob(os.path.join(keyframe_dir, "data", "left", "*.png")))
            image_right_paths = sorted(glob.glob(os.path.join(keyframe_dir, "data", "right", "*.png")))
            depth_paths = sorted(glob.glob(os.path.join(keyframe_dir, "data", "scene_points", "*.tiff")))
            # depth_paths = sorted(glob.glob(os.path.join(keyframe_dir, "data", "scene_points_left", "*.npy")))
            pose_paths = sorted(glob.glob(os.path.join(keyframe_dir, "data", "frame_data", "*.json")))
            assert len(image_left_paths) == len(image_right_paths) == len(depth_paths) == len(pose_paths)
            data_paths_dict['images_left'].extend(image_left_paths)
            data_paths_dict['images_right'].extend(image_right_paths)
            data_paths_dict['depths'].extend(depth_paths)
            data_paths_dict['poses'].extend(pose_paths)
        return data_paths_dict

    def get_color(self, indices, side, do_flip):
        colors = []
        for index in indices:
            color = self.loader(self.data_paths_dict[f'images_{self.side_map[side]}'][index])
            if do_flip:
                color = color.transpose(Image.FLIP_LEFT_RIGHT)
            colors.append(self.to_tensor(color))
        return torch.stack(colors, dim=0) # [B, C, H, W]

    def get_depth(self, indices, do_flip):
        depths = []
        for index in indices:
            depth_gt = cv2.imread(self.data_paths_dict['depths'][index], 3).astype(np.float32)
            depth_gt = depth_gt[:, :, 0]
            depth_gt = depth_gt[0:1024, :]
            # depth_gt = np.load(self.data_paths_dict['depths'][index]).astype(np.float32)
            if do_flip:
                depth_gt = np.fliplr(depth_gt)
            depths.append(depth_gt[None])
        return torch.from_numpy(np.stack(depths, axis=0))
    
    def get_pose(self, indices):
        poses = []
        for index in indices:
            pose_path = self.data_paths_dict['poses'][index]
            with open(pose_path, 'r') as path:
                data = json.load(path)
                pose = np.linalg.pinv(np.array(data['camera-pose']))
            poses.append(pose)
        return torch.from_numpy(np.stack(poses, axis=0))
