from __future__ import absolute_import, division, print_function

import glob
import os
import random
import numpy as np
from tqdm import tqdm
import cv2
import imageio
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
        
def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_images(image_dir, max_length=None, max_res=-1):
    images = []
    img_files = sorted(os.listdir(image_dir), key=lambda x: int(x[:-4]))
    if max_length is not None:
        img_files = img_files[:max_length]
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

def read_depths(depth_dir, max_length=None):
    depths = []
    dep_files = sorted(os.listdir(depth_dir))
    if max_length is not None:
        dep_files = dep_files[:max_length]
    for file in tqdm(dep_files, desc="Reading depths..."):
        depth_path = os.path.join(depth_dir, file)
        if file.endswith('.tiff'):
            depth = cv2.imread(depth_path, 3).astype(np.float32)[0:1024, :, 0]
        elif file.endswith('.npy'):
            depth = np.load(depth_path).astype(np.float32)
        elif file.endswith('.png'):
            depth = np.array(Image.open(depth_path)).astype(np.float32)
        depths.append(depth)
    return np.stack(depths, axis=0) # [B, H, W]

def load_sequence(data_path, filename, max_length=None):
    keyframe_dir = os.path.join(data_path, filename)
    colors = read_images(os.path.join(keyframe_dir, "image01"), max_length)
    depths = read_depths(os.path.join(keyframe_dir, "depth01"), max_length)
    assert len(colors) == len(depths)
    return colors, depths

class HamlynVideos(data.Dataset):
    def __init__(self, data_path, filenames, pred_root=None, max_length=None):
        super(HamlynVideos, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.max_length = max_length
        self.pred_root = pred_root
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.pred_root is None:
            colors, depths = load_sequence(self.data_path, filename, self.max_length)
            return {
                "colors": colors,
                "depths": depths,
                "filename": filename
            }
        else:
            keyframe_dir = os.path.join(self.data_path, filename)
            depths = read_depths(os.path.join(keyframe_dir, "depth01"), self.max_length)
            pred_dir = os.path.join(self.pred_root, filename)
            pred_depths = read_depths(os.path.join(pred_dir, "depth"), self.max_length)
            assert len(depths) == len(pred_depths)
            return {
                "depths": depths,
                "pred_depths": pred_depths,
                "filename": filename
            }
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class HamlynDataset(data.Dataset):
    """Superclass for monocular dataloaders

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
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super(HamlynDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

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
                                               interpolation=self.interp)
        
        self.long_rectified_files = ["rectified14", "rectified14", "rectified14", "rectified14"]
        self.scans = []
        self.rectified_files = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        self.rectified_files.sort()
        self.long_rectified_files = self.rectified_files[7:]
        self.sequence_len = np.zeros([len(self.rectified_files)])
        for i, rectified_file in enumerate(self.rectified_files):
            image01_paths = os.path.join(rectified_file, "image01", "*.jpg")
            seq_image01_paths = glob.glob(image01_paths)
            seq_image01_paths.sort()
            for seq_image01_path in seq_image01_paths:
                filename = os.path.basename(seq_image01_path)
                seq_image02_path = os.path.join(rectified_file, "image02", filename)
                seq_depth01_path = os.path.join(rectified_file, "depth01", filename[:-4]+".png")
                seq_depth02_path = os.path.join(rectified_file, "depth02", filename[:-4]+".png")
                
                if os.path.exists(seq_image01_path) and os.path.exists(seq_image02_path) and os.path.exists(seq_depth01_path) and os.path.exists(seq_depth02_path):
                    sequence = int(rectified_file[-2:])
                    self.sequence_len[i] += 1
                    self.scans.append(
                        {
                            "image01": seq_image01_path,
                            "image02": seq_image02_path,
                            "depth01": seq_depth01_path,
                            "depth02": seq_depth02_path,
                            "sequence": sequence,
                            "index":int(filename[:-4]),
                            "length": len(seq_image01_paths),
                        })
        print("Prepared Hamlyn dataset with %d sets of left & right images, left & right depths." % (len(self.scans)))
        self.box = (180, 0, 590, 288)
    def __len__(self):
        return len(self.scans)
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    def get_color(self, path, do_flip):
        color = self.loader(path)
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, path, do_flip):

        depth_gt = np.array(Image.open(path))
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
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
        scan = self.scans[index]
        sequence = scan["sequence"]
        inputs = {}

        inputs["sequence"] = scan["sequence"]
        inputs["index"] = scan["index"]
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        inputs[("color", 0, 0)] = self.get_color(scan["image01"], do_flip)
        inputs["depth_gt"] = self.get_depth(scan["depth01"], do_flip)
        
        if sequence > 13:
            inputs[("color", 0, 0)] = inputs[("color", 0, 0)].crop(self.box)
            inputs["depth_gt"] = inputs["depth_gt"][:,180:590]
        inputs[("color", 0, 0)] = self.resize[0](inputs[("color", 0, 0)])
        inputs[("color", 0, 0)] = self.to_tensor(inputs[("color", 0, 0)])
        
        return inputs
# if __name__ == "__main__":
#     ds = HamlynDataset(data_path='/mnt/data-hdd2/Beilei/Dataset/Hamlyn', height=256, width=320, frame_idxs=[0, -1, 1], num_scales=4, is_train=True)
    
#     test = ds[0]