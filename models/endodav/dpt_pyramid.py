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
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dpt_temporal import DPTHeadTemporal
from .layers import HeadDepth
from easydict import EasyDict


class DPTHeadPyramid(DPTHeadTemporal):
    def __init__(self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        inv_sigmoid=False,
        disable_conv_head=False,
        out_sigmoid=False
    ):
        super().__init__(in_channels, features, use_bn, out_channels, use_clstoken, num_frames, pe)

        self.disable_conv_head = disable_conv_head
        self.out_sigmoid = out_sigmoid
        if not disable_conv_head:
            del self.scratch.output_conv1
            del self.scratch.output_conv2
            self.conv_depth_1 = HeadDepth(features)
            self.conv_depth_2 = HeadDepth(features)
            self.conv_depth_3 = HeadDepth(features)
            self.conv_depth_4 = HeadDepth(features)
            self.sigmoid = nn.Sigmoid()
            self.inv_sigmoid = inv_sigmoid
        if out_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, out_features, patch_h, patch_w, frame_length):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w)).contiguous()

            B, T = x.shape[0] // frame_length, frame_length
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        B, T = layer_1.shape[0] // frame_length, frame_length

        layer_3 = self.motion_modules[0](layer_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        layer_4 = self.motion_modules[1](layer_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_4 = self.motion_modules[2](path_4.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_3 = self.motion_modules[3](path_3.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4), None, None).permute(0, 2, 1, 3, 4).flatten(0, 1)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if self.disable_conv_head:
            output = self.scratch.output_conv1(path_1)
            output = F.interpolate(
                output, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True
            )
            output = self.scratch.output_conv2(output)
            out = {("disp", 0): output}
            out[("disp", 1)] = F.interpolate(out[("disp", 0)], scale_factor=0.5, mode="bilinear", align_corners=True)
            out[("disp", 2)] = F.interpolate(out[("disp", 1)], scale_factor=0.5, mode="bilinear", align_corners=True)
            out[("disp", 3)] = F.interpolate(out[("disp", 2)], scale_factor=0.5, mode="bilinear", align_corners=True)
            if self.out_sigmoid:
                out[("disp", 0)] = self.sigmoid(out[("disp", 0)])
                out[("disp", 1)] = self.sigmoid(out[("disp", 1)])
                out[("disp", 2)] = self.sigmoid(out[("disp", 2)])
                out[("disp", 3)] = self.sigmoid(out[("disp", 3)])
        else:
            out = {}
            sig_sign = -1 if self.inv_sigmoid else 1
            out[("disp", 3)] = self.sigmoid(sig_sign*self.conv_depth_4(path_4))
            out[("disp", 2)] = self.sigmoid(sig_sign*self.conv_depth_3(path_3))
            out[("disp", 1)] = self.sigmoid(sig_sign*self.conv_depth_2(path_2))
            out[("disp", 0)] = self.sigmoid(sig_sign*self.conv_depth_1(path_1))
        
        # print('out shape', out[("disp", 0)].shape)

        return out