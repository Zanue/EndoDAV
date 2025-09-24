from __future__ import absolute_import, division, print_function

import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import scipy.stats as st

from utils.layers import disp_to_depth, transformation_from_parameters
from utils.utils import readlines, compute_errors, dump_xyz, dump_r, compute_ate, compute_re
from utils.eval_utils import tae, tas, median_scaling, align_shift_and_scale, save_npy, save_video, vis_pose_sq
from options import MonodepthOptions
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac
import models.endodav as endodav

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
cv2.ocl.setUseOpenCL(False)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    
    # pose related
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
    intrinsics_decoder_path = os.path.join(opt.load_weights_folder, "intrinsics_head.pth")
    pose_encoder = encoders.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))
    if opt.learn_intrinsics:
        intrinsics_decoder = decoders.IntrinsicsHead(pose_encoder.num_ch_enc)
        intrinsics_decoder.load_state_dict(torch.load(intrinsics_decoder_path))
        intrinsics_decoder.cuda()
        intrinsics_decoder.eval()
    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    if opt.ext_disp_to_eval is None:
        if not opt.model_type == 'depthanything':
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)

            print("-> Loading weights from {}".format(opt.load_weights_folder))
        else:
            print("Evaluating Depth Anything model")

        if opt.model_type == 'endodac' or opt.model_type == 'endodav':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)
        elif opt.model_type == 'afsfm':
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path)

        if opt.disable_residual_block:
            opt.residual_block_indexes = []

        if opt.model_type == 'endodav':
            depth_model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            depther = endodav.endodav(
                **depth_model_configs[opt.encoder], r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=(224,280), pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token,
                inv_sigmoid=opt.inv_sigmoid, temporal_lora=opt.temporal_lora,
                disable_conv_head=opt.disable_conv_head, out_sigmoid=opt.out_sigmoid)
            model_dict = depther.state_dict()

            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'endodac':
            backbone_size_config = {
                'vits': 'small',
                'vitb': 'base',
                'vitl': 'large',
            }
            depther = endodac.endodac(
                backbone_size = backbone_size_config[opt.encoder], r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=(224,280), pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token,
                pre_norm=opt.pre_norm, inv_sigmoid=opt.inv_sigmoid,
                disable_conv_head=opt.disable_conv_head)
            model_dict = depther.state_dict()
            
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda()
            depther.eval()
        elif opt.model_type == 'afsfm':
            encoder = encoders.ResnetEncoder(opt.num_layers, False)
            depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))
            depther = lambda image: depth_decoder(encoder(image))
            encoder.cuda()
            encoder.eval()
            depth_decoder.cuda()
            depth_decoder.eval()
    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
    
    if opt.eval_split == 'scared_video':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_files.txt"))
        dataset = datasets.SCAREDVideos(opt.data_path, filenames)
    elif opt.eval_split == 'endovis':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                        opt.height, opt.width,
                                        [0], 4, is_train=False)
    elif opt.eval_split == 'hamlyn':
        dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                            [0], 4, is_train=False)
    elif opt.eval_split == 'c3vd':
        dataset = datasets.C3VDDataset(opt.data_path, opt.height, opt.width,
                                            [0], 4, is_train=False)
        MAX_DEPTH = 100

    # dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=0,
    #                         pin_memory=True, drop_last=False)
    dataloader = dataset

    if opt.visualize_depth:
        eval_dir = os.path.join(opt.load_weights_folder, 'eval', opt.eval_split)
        os.makedirs(eval_dir, exist_ok=True)

    inference_times = []
    
    errors = []
    errors_temp = []
    ratios = []
    t_gts, s_gts, t_preds, s_preds = [], [], [], []
    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            # input_colors, gt_depths, poses, Ks = data['colors'].numpy()[0], data['depths'].numpy()[0], data['poses'].numpy()[0], data['Ks'].numpy()[0]
            input_colors, gt_depths, poses, Ks = data['colors'], data['depths'], data['poses'], data['Ks'] # pose: w2c
            split, sequence, keyframe = data['filename'].split('/')
            time_start = time.time()
            output_disp = depther.infer_video_depth(input_colors)
            inference_time = time.time() - time_start

            _, pred_depths = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
            inference_times.append(inference_time)
            # split, sequence, keyframe = data['filename'][0].split('/')

            if opt.depth_align == 'scale':
                pred_depths, ratio = median_scaling(gt_depths, pred_depths)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
            elif opt.depth_align == 'scale_shift':
                pred_depths, t_gt, s_gt, t_pred, s_pred = align_shift_and_scale(gt_depths, pred_depths)
                t_gts.append(t_gt)
                s_gts.append(s_gt)
                t_preds.append(t_pred)
                s_preds.append(s_pred)

            if opt.visualize_depth:
                seq_dir = os.path.join(eval_dir, sequence, keyframe)
                depth_dir = os.path.join(seq_dir, "depth")
                os.makedirs(seq_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
                save_video(input_colors, pred_depths, os.path.join(seq_dir, "vis.mp4"))
                save_npy(pred_depths, depth_dir)
            
            prevs = {
                'pred_depth': None,
                'gt_depth': None,
                'mask': None,
                'img2lidar': None
            }
            for (pred_depth, gt_depth, pose, K) in zip(pred_depths, gt_depths, poses, Ks):
                valid_mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                error = compute_errors(gt_depth, pred_depth, valid_mask)
                if not np.isnan(error).all():
                    errors.append(error)
                img2lidar = np.linalg.inv(K @ pose)
                if prevs['pred_depth'] is not None:
                    error_tae = tae(prevs['pred_depth'], prevs['mask'], prevs['img2lidar'], \
                        pred_depth, valid_mask, img2lidar) * 100.
                    error_tas = tas(prevs['pred_depth'], prevs['mask'], prevs['img2lidar'], \
                        pred_depth, valid_mask, img2lidar)
                    errors_temp.append([error_tae, error_tas])
                prevs['pred_depth'], prevs['gt_depth'], prevs['mask'], prevs['img2lidar'] = pred_depth, gt_depth, valid_mask, img2lidar

            # log
            txt_str = "{}_{}\n".format(sequence, keyframe)
            if opt.depth_align == 'scale':
                txt_str += " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(ratio, 0.)
                print(txt_str)
            elif opt.depth_align == 'scale_shift':
                txt_str += " Aligning shift and scale | t_gt: {:0.3f} | s_gt: {:0.3f} | t_pred: {:0.3f} | s_pred: {:0.3f}\n".format(
                    t_gt, s_gt, t_pred, s_pred)
                print(txt_str)
            
            error = np.array(error)
            txt_str += ("{:>11}      | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "tae", "tas")
            txt_str += "\nmean:" + ("&{: 12.3f}      " * 9).format(*(error.tolist()+[error_tae, error_tas])) + "\\\\\n\n"
            print(txt_str)
            with open(os.path.join(eval_dir, 'results_eval.txt'), 'a') as f:
                f.write(txt_str)

            # eval pose
            pred_poses, pred_intrinsics = [], []
            for idx in range(len(poses)-1):
                color_0, color_1 = input_colors[idx].astype(np.float32)/255., input_colors[idx+1].astype(np.float32)/255.
                color_0 = torch.from_numpy(color_0).unsqueeze(0).permute(0,3,1,2).cuda()
                color_1 = torch.from_numpy(color_1).unsqueeze(0).permute(0,3,1,2).cuda()
                all_color_aug = torch.cat([color_1, color_0], 1)
                features = [pose_encoder(all_color_aug)]
                axisangle, translation, intermediate_feature = pose_decoder(features)
                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                if opt.learn_intrinsics:
                    cam_K = intrinsics_decoder(
                            intermediate_feature, opt.width, opt.height)
                    pred_intrinsics.append(cam_K[:,:3,:3].cpu().numpy())
            pred_poses = np.concatenate(pred_poses)
            if opt.learn_intrinsics:
                pred_intrinsics = np.concatenate(pred_intrinsics)

            gt_local_poses = []
            for i in range(0, len(poses) - 1):
                data_0 = np.linalg.inv(poses[i])
                data_1 = poses[i+1]
                T = (data_1 @ data_0).astype(np.float32)
                gt_local_poses.append(T)
            gt_local_poses = np.array(gt_local_poses)

            ates_1 = []
            res_1 = []
            track_length = 5
            for i in range(0, len(poses) - 1):
                local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
                gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
                local_rs = np.array(dump_r(pred_poses[i:i + track_length - 1]))
                gt_rs = np.array(dump_r(gt_local_poses[i:i + track_length - 1]))
                ates_1.append(compute_ate(gt_local_xyzs, local_xyzs))
                res_1.append(compute_re(local_rs, gt_rs))
            cls_1 = st.t.interval(alpha=0.95, df=len(ates_1)-1, loc=np.mean(ates_1), scale=st.sem(ates_1))
            cls_1 = np.array(cls_1)

            sq_str = "\n   sq Trajectory error: {:0.4f}, std: {:0.4f}, 95% cls: [{:0.4f}, {:0.4f}]\n".format(np.mean(ates_1), np.std(ates_1), cls_1[0], cls_1[1])
            sq_str += "\n   sq Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res_1), np.std(res_1))
            print(sq_str)
            if opt.learn_intrinsics:
                fx_mean, fx_std = np.mean(pred_intrinsics[:,0,0]) / opt.width, np.std(pred_intrinsics[:,0,0]) / opt.width
                fy_mean, fy_std = np.mean(pred_intrinsics[:,1,1]) / opt.height, np.std(pred_intrinsics[:,1,1]) / opt.height
                cx_mean, cx_std = np.mean(pred_intrinsics[:,0,2]) / opt.width, np.std(pred_intrinsics[:,0,2]) / opt.width
                cy_mean, cy_std = np.mean(pred_intrinsics[:,1,2]) / opt.height, np.std(pred_intrinsics[:,1,2]) / opt.height
                intrinsics_str = "\n   fx: {:0.4f}, std: {:0.4f}\n".format(fx_mean, fx_std)
                intrinsics_str += "\n   fy: {:0.4f}, std: {:0.4f}\n".format(fy_mean, fy_std)
                intrinsics_str += "\n   cx: {:0.4f}, std: {:0.4f}\n".format(cx_mean, cx_std)
                intrinsics_str += "\n   cy: {:0.4f}, std: {:0.4f}\n".format(cy_mean, cy_std)
                print(intrinsics_str)

            with open(os.path.join(eval_dir, "pose_eval.txt"), "a") as f:
                f.write(sq_str)
                if opt.learn_intrinsics:
                    f.write(intrinsics_str + "\n")

            if opt.visualize_depth:
                vis_pose_dir = os.path.join(eval_dir, "pose")
                os.makedirs(vis_pose_dir, exist_ok=True)
                vis_pose_sq(pred_poses, gt_local_poses, save_path=os.path.join(vis_pose_dir, f"{sequence}_{keyframe}.png"))
                

    if opt.depth_align == 'scale':
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    elif opt.depth_align == 'scale_shift':
        t_gts = np.array(t_gts)
        s_gts = np.array(s_gts)
        t_preds = np.array(t_preds)
        s_preds = np.array(s_preds)
        print(" Aligning shift and scale | t_gt: {:0.3f} | s_gt: {:0.3f} | t_pred: {:0.3f} | s_pred: {:0.3f}".format(
            np.mean(t_gts), np.mean(s_gts), np.mean(t_preds), np.mean(s_preds)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    cls = []
    for i in range(len(mean_errors)):
        cl = st.t.interval(alpha=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)
    errors_temp = np.array(errors_temp)
    mean_errors_temp = np.mean(errors_temp, axis=0)
    cls_temp = []
    for i in range(len(mean_errors_temp)):
        cl = st.t.interval(alpha=0.95, df=len(errors_temp)-1, loc=mean_errors_temp[i], scale=st.sem(errors_temp[:,i]))
        cls_temp.append(cl[0])
        cls_temp.append(cl[1])
    cls_temp = np.array(cls_temp)
    txt_str = ""
    txt_str += ("{:>11}      | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "tae", "tas")
    txt_str += "\nmean:" + ("&{: 12.3f}      " * 9).format(*mean_errors.tolist(), *mean_errors_temp.tolist()) + "\\\\"
    txt_str += "\ncls: " + ("& [{: 6.3f}, {: 6.3f}] " * 9).format(*cls.tolist(), *cls_temp.tolist()) + "\\\\"
    txt_str += "\naverage inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000)
    print(txt_str)
    with open(os.path.join(opt.load_weights_folder, 'eval', opt.eval_split, "results_eval.txt"), 'a') as f:
        f.write(txt_str)
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
