# Copyright (2025) Bytedance Ltd. and/or its affiliates 

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
import cv2
from tqdm import tqdm
import numpy as np
import gc

from .dinov2 import DINOv2
from .dpt_pyramid import DPTHeadPyramid
import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from models.backbones.mylora import Linear_SSB as Linear_SSB
from models.backbones.mylora import DashLinear as DashLinear
from .util.transform import Resize, NormalizeImage, PrepareForNet
from .layers import mark_only_part_as_trainable, _make_scratch, _make_fusion_block

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
# INFER_LEN = 32
# OVERLAP = 10
# KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
# INTERP_LEN = 8

# INFER_LEN = 16
# OVERLAP = 6
# KEYFRAMES = [0,6,12,13,14,15]
# INTERP_LEN = 4

INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [6,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class endodav(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False,
        num_frames=32,
        pe='ape', 
        # endodav settings
        r=4, 
        image_shape=(224,280), 
        lora_type="lora",
        pretrained_path=None,
        residual_block_indexes=[],
        include_cls_token=True,
        inv_sigmoid=False,
        temporal_lora=False,
        disable_conv_head=False,
        out_sigmoid=False
    ):
        super(endodav, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }
        self.backbone = {
            "vits": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "vitl": backbones.vits.vit_large(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token),
        }

        # self.normalize = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.encoder = encoder
        # self.pretrained = DINOv2(model_name=encoder)
        self.pretrained = self.backbone[encoder]

        self.head = DPTHeadPyramid(self.pretrained.embed_dim, features, use_bn, \
                                   out_channels=out_channels, use_clstoken=use_clstoken, \
                                   num_frames=num_frames, pe=pe, inv_sigmoid=inv_sigmoid, \
                                   disable_conv_head=disable_conv_head, out_sigmoid=out_sigmoid)

        # lora
        self.image_shape = image_shape
        self.r = r
        if lora_type != "none":
            for t_layer_i, blk in enumerate(self.pretrained.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=2*self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=2*self.r)
                elif lora_type == 'ssb':
                    blk.mlp.fc1 = Linear_SSB(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = Linear_SSB(mlp_hidden_features, mlp_out_features, r=self.r)
                elif lora_type == 'dash':
                    blk.mlp.fc1 = DashLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=2*self.r)
                    blk.mlp.fc2 = DashLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=2*self.r)
            if temporal_lora:
                for m_layer_i, tm in enumerate(self.head.motion_modules):
                    # mlp_in_features = tm.temporal_transformer.proj_out.in_features
                    # mlp_out_features = tm.temporal_transformer.proj_out.out_features
                    # if lora_type == "dvlora":
                    #     tm.temporal_transformer.proj_out = DVLinear(mlp_in_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                    # elif lora_type == "lora":
                    #     tm.temporal_transformer.proj_out = LoraLinear(mlp_in_features, mlp_out_features, r=self.r, lora_alpha=2*self.r)
                    for t_layer_i, blk in enumerate(tm.temporal_transformer.transformer_blocks):
                        mlp_in_features = blk.ff.net[2].in_features
                        mlp_out_features = blk.ff.net[2].out_features
                        if lora_type == "dvlora":
                            blk.ff.net[2] = DVLinear(mlp_in_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                        elif lora_type == "lora":
                            blk.ff.net[2] = LoraLinear(mlp_in_features, mlp_out_features, r=self.r, lora_alpha=2*self.r)
                        elif lora_type == 'ssb':
                            blk.ff.net[2] = Linear_SSB(mlp_in_features, mlp_out_features, r=self.r)
                        elif lora_type == 'dash':
                            blk.ff.net[2] = DashLinear(mlp_in_features, mlp_out_features, r=self.r, lora_alpha=2*self.r)

        if pretrained_path is not None:
            print("load pretrained weight from {}\n".format(pretrained_path))
            pretrained_path = os.path.join(pretrained_path, "video_depth_anything_{}.pth".format(self.encoder))
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict, strict=False)
            
        mark_only_part_as_trainable(self.pretrained)
        mark_only_part_as_trainable(self.head)
        mark_only_part_as_trainable(self.head.motion_modules, is_trainable=False)

    def forward(self, x):
        # endo scenes do not need high res input
        B, T, C, H, W = x.shape
        x_resize = torch.nn.functional.interpolate(x.flatten(0,1), size=self.image_shape, mode="bilinear", align_corners=True)
        # x_resize = x.flatten(0,1)
        x_norm = self.normalize(x_resize)
        
        patch_h, patch_w = x_norm.shape[-2] // 14, x_norm.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x_norm, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        disp = self.head(features, patch_h, patch_w, T)
        return disp # return shape [B*T, 1, H, W]
    
    def infer_video_depth(self, frames, input_size=518, device='cuda'):
        frame_height, frame_width = frames[0].shape[:2]
        # ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        # if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
        #     input_size = int(input_size * 1.777 / ratio)
        #     input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=self.image_shape[1],
                height=self.image_shape[0],
                # width=input_size,
                # height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len
        
        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                depth = self.forward(cur_input) # depth shape: [1, T, H, W]
                depth = depth[("disp", 0)]

            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])
                # scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                #                                        np.concatenate(ref_align),
                #                                        np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                scale, shift = compute_scale_and_shift(np.concatenate(post_depth_list),
                                                       np.concatenate(pre_depth_list),
                                                       np.concatenate(np.ones_like(pre_depth_list)==1))
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)
            
        depth_list = depth_list_aligned
            
        return np.stack(depth_list[:org_video_len], axis=0) # shape: [T, H, W]
        