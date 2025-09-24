from __future__ import absolute_import, division, print_function

import time
import json
from tqdm import tqdm
import datasets
import scipy.stats as st
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac
import models.endodav as endodav
import numpy as np
import torch.optim as optim
from torch import nn

from utils.utils import *
from utils.layers import *
from utils.eval_utils import median_scaling, align_shift_and_scale, save_npy, save_video, tae, tas, vis_pose_sq
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_type)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # 字典
        self.parameters_to_train = []  # 列表
        self.parameters_to_train_0 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")

        
        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        if self.opt.disable_residual_block:
            self.opt.residual_block_indexes = []

        if self.opt.model_type == 'endodav':
            depth_model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            }
            self.models["depth_model"] = endodav.endodav(
                **depth_model_configs[self.opt.encoder], r=self.opt.lora_rank, lora_type=self.opt.lora_type,
                image_shape=(224,280), pretrained_path=self.opt.pretrained_path,
                residual_block_indexes=self.opt.residual_block_indexes,
                include_cls_token=self.opt.include_cls_token,
                inv_sigmoid=self.opt.inv_sigmoid, temporal_lora=self.opt.temporal_lora, 
                disable_conv_head=self.opt.disable_conv_head,
                out_sigmoid=self.opt.out_sigmoid)
        elif self.opt.model_type == 'endodac':
            backbone_size_config = {
                'vits': 'small',
                'vitb': 'base',
                'vitl': 'large',
            }
            self.models["depth_model"] = endodac.endodac(
                backbone_size = backbone_size_config[self.opt.encoder], r=self.opt.lora_rank, lora_type=self.opt.lora_type,
                image_shape=(224,280), pretrained_path=self.opt.pretrained_path,
                residual_block_indexes=self.opt.residual_block_indexes,
                include_cls_token=self.opt.include_cls_token,
                pre_norm=self.opt.pre_norm, inv_sigmoid=self.opt.inv_sigmoid,
                disable_conv_head=self.opt.disable_conv_head)
        self.parameters_to_train += list(filter(lambda p: p.requires_grad, self.models["depth_model"].parameters()))
        self.tune_temporal = False
        self.tune_depth = False if self.opt.tune_depth_interval > 0 else True

        self.models["position_encoder"] = encoders.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.parameters_to_train_0 += list(self.models["position_encoder"].parameters())

        self.models["position"] = decoders.PositionDecoder(
            self.models["position_encoder"].num_ch_enc, self.opt.scales)
        self.parameters_to_train_0 += list(self.models["position"].parameters())

        self.models["transform_encoder"] = encoders.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=2)  # 18
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = decoders.TransformDecoder(
            self.models["transform_encoder"].num_ch_enc, self.opt.scales)
        self.parameters_to_train += list(self.models["transform"].parameters())

        if self.use_pose_net:

            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = encoders.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = decoders.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = decoders.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = decoders.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.parameters_to_train += list(self.models["pose"].parameters())
            
            if self.opt.learn_intrinsics:
                self.models['intrinsics_head'] = decoders.IntrinsicsHead(self.models["pose_encoder"].num_ch_enc)
                self.parameters_to_train += list(self.models['intrinsics_head'].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = decoders.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        self.model_optimizer_0 = optim.Adam(self.parameters_to_train_0, 1e-4)
        self.model_lr_scheduler_0 = optim.lr_scheduler.StepLR(
            self.model_optimizer_0, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.use_dp:
            self.set_DataParallel()
        self.set_model2device()

        print("Training model named:\n  ", self.opt.model_type)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        if self.opt.model_type == 'endodac':
            datasets_dict = {"scared_video": datasets.SCAREDRAWDataset}
            self.opt.split = 'endovis'
        elif self.opt.model_type == 'endodav':
            datasets_dict = {"scared_video": datasets.SCAREDRAWVideoDataset}
            self.opt.split = 'scared_video'
        else:
            raise NotImplementedError
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        fpath = os.path.join(os.path.dirname(__file__), "splits", 'scared_video', "{}_files.txt")
        test_filenames = readlines(fpath.format("val"))
        img_ext = '.png'  

        self.train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, T=self.opt.T, frame_max_interval=self.opt.frame_max_interval)
        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, 
            # True,
            True if self.opt.model_type == 'endodav' else False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, T=self.opt.T)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=1, pin_memory=True, drop_last=True)
        # test_dataset = self.dataset(
        #     self.opt.data_path, test_filenames, self.opt.height, self.opt.width,
        #     self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, T=self.opt.T)
        # self.test_loader = DataLoader(
        #     test_dataset, 1, False,
        #     num_workers=1, pin_memory=True, drop_last=True)
        test_dataset = datasets.SCAREDVideos(self.opt.data_path, test_filenames)
        # self.test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0,
        #                     pin_memory=True, drop_last=False)
        self.test_loader = test_dataset
        self.val_iter = iter(self.val_loader)

        num_train_samples = len(self.train_dataset)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size*self.opt.T, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size*self.opt.T, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size*self.opt.T, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rmse", "de/log_rmse", "da/a1", "da/a2", "da/a3", 
            "temp/tae", "temp/tas"]

        # gt_path = os.path.join(splits_dir, self.opt.eval_split, "gt_depths.npz")
        # self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
        
        print("Using split:\n  ", self.opt.split)
        # print("There are {:d} training items, {:d} validation items and {:d} testing items\n".format(
        #     len(train_dataset), len(val_dataset), len(test_dataset)))

        self.save_opts()
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for name, param in self.models["depth_model"].named_parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad == False:
                NonTrainable_params += mulValue
                # print(name)
            else:
                Trainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        print(f'Trainable params ratio: {100 * Trainable_params / Total_params}%')
        print('scales: ', self.opt.scales)
        print('learn intrinsics: ', self.opt.learn_intrinsics)

    def set_DataParallel(self):
        for key in self.models.keys():
            self.models[key] = nn.DataParallel(self.models[key])

    def set_model2device(self):
        for key in self.models.keys():
            self.models[key].to(self.device)

    def set_train_0(self):
        """Convert all models to training mode
        """
        for param in self.models["position_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["position"].parameters():
            param.requires_grad = True

        for param in self.models["depth_model"].parameters():
            param.requires_grad = False
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["pose"].parameters():
            param.requires_grad = False
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["transform"].parameters():
            param.requires_grad = False
        if self.opt.learn_intrinsics:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = False
            
        self.models["position_encoder"].train()
        self.models["position"].train()

        self.models["depth_model"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].eval()

    def set_train(self):
        """Convert all models to training mode
        """
        for param in self.models["position_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["position"].parameters():
            param.requires_grad = False
        
        tune_depth, tune_pose = True, True
        if self.opt.tune_depth_interval > 0:
            tune_depth = (self.step % (2*self.opt.tune_depth_interval)) >= self.opt.tune_depth_interval
            tune_pose = not tune_depth
        self.tune_depth = tune_depth
        
        for name, param in self.models["depth_model"].named_parameters():
            if "seed_" not in name:
                param.requires_grad = tune_depth

        warm_up = True
        if self.opt.lora_type == 'dvlora' and self.step > self.opt.warm_up_step:
            warm_up = False
        
        tune_spatial, tune_temporal = True, False
        if self.opt.temporal_lora:
            total_tune_ival = self.opt.tune_spatial_interval + self.opt.tune_temporal_interval
            if (self.step % total_tune_ival) >= self.opt.tune_spatial_interval:
                tune_spatial, tune_temporal = False, True
        endodav.mark_only_part_as_trainable(self.models["depth_model"], warm_up=warm_up, is_trainable=tune_spatial and tune_depth, train_output_conv=self.opt.train_output_conv)
        if self.opt.temporal_lora:
            endodav.mark_only_part_as_trainable(self.models["depth_model"].head.motion_modules, warm_up=warm_up, is_trainable=tune_temporal and tune_depth)
        self.tune_temporal = tune_temporal

        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = tune_pose
        for param in self.models["pose"].parameters():
            param.requires_grad = tune_pose
        for param in self.models["transform_encoder"].parameters():
            param.requires_grad = tune_pose
        for param in self.models["transform"].parameters():
            param.requires_grad = tune_pose
        if self.opt.learn_intrinsics:
            for param in self.models["intrinsics_head"].parameters():
                param.requires_grad = tune_pose

        self.models["position_encoder"].eval()
        self.models["position"].eval()

        self.models["depth_model"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["transform_encoder"].train()
        self.models["transform"].train()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["depth_model"].eval()
        self.models["transform_encoder"].eval()
        self.models["transform"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        if self.opt.learn_intrinsics:
            self.models["intrinsics_head"].eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 1
        # rmse, a1 = self.run_epoch_eval()
        self.start_time = time.time()
        for self.epoch in tqdm(range(1, self.opt.num_epochs+1), desc=f'Training...'):
            self.run_epoch()
            if self.epoch == 1:
                rmse, a1 = self.run_epoch_eval()
                self.save_model(mode='epoch')
            else:
                rmse_new, a1_new = self.run_epoch_eval()
                if rmse_new < rmse:
                    rmse = rmse_new
                    # a1 = a1_new
                    self.save_model(mode='epoch')
            self.save_model(mode='last')
    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")

        for batch_idx, inputs in tqdm(enumerate(self.train_loader), desc=f'Epoch {self.epoch}'):

            before_op_time = time.time()
            
            # flatten Batch_size and Time dims
            if self.opt.T > -1:
                for key in inputs.keys():
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].flatten(0, 1)
                        # print('{}, shape: {}'.format(key, inputs[key].shape))

            # if randomly train
            if self.opt.random_train:
                if not self.tune_depth:
                    self.train_dataset.random_train = True
                else:
                    self.train_dataset.random_train = False

            # position
            self.set_train_0()
            _, losses_0 = self.process_batch_0(inputs)
            self.model_optimizer_0.zero_grad()
            losses_0["loss"].backward()
            self.model_optimizer_0.step()

            # depth, pose, transform
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            
            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1
            
        self.model_lr_scheduler.step()
        self.model_lr_scheduler_0.step()

    @torch.no_grad()
    def run_epoch_eval(self):
        """Run a single epoch of evaluation
        """

        print("Evaluating")
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 150
        
        self.set_eval()
        errors = []
        errors_temp = []
        ratios = []
        t_gts, s_gts, t_preds, s_preds = [], [], [], []
        model_folder = os.path.join(self.log_path, "models")
        save_folder = os.path.join(model_folder, "weights_{}".format(self.epoch), 'eval', self.opt.eval_split)
        os.makedirs(save_folder, exist_ok=True)
        
        for batch_idx, data in tqdm(enumerate(self.test_loader), desc='Eval...'):
            # input_colors, gt_depths = data['colors'][0].numpy(), data['depths'][0].numpy()
            input_colors, gt_depths, poses, Ks = data['colors'], data['depths'], data['poses'], data['Ks']
            split, sequence, keyframe = data['filename'].split('/')

            # eval depth
            output_disp = self.models["depth_model"].infer_video_depth(input_colors)
            _, pred_depths = disp_to_depth(output_disp, self.opt.min_depth, self.opt.max_depth)
            
            if self.opt.depth_align == 'scale':
                pred_depths, ratio = median_scaling(gt_depths, pred_depths)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
            elif self.opt.depth_align == 'scale_shift':
                pred_depths, t_gt, s_gt, t_pred, s_pred = align_shift_and_scale(gt_depths, pred_depths)
                t_gts.append(t_gt)
                s_gts.append(s_gt)
                t_preds.append(t_pred)
                s_preds.append(s_pred)

            if self.opt.visualize_depth:
                seq_dir = os.path.join(save_folder, sequence, keyframe)
                depth_dir = os.path.join(seq_dir, "depth")
                os.makedirs(seq_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)
                save_video(input_colors, pred_depths, os.path.join(seq_dir, "vis.mp4"))
                save_npy(pred_depths, depth_dir)
                save_npy(pred_depths, depth_dir)

            prevs = {
                'pred_depth': None,
                'gt_depth': None,
                'mask': None,
                'img2lidar': None
            }
            for (pred_depth, gt_depth, pose, K) in zip(pred_depths, gt_depths, poses, Ks):
                valid_mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                pred_depth *= self.opt.pred_depth_scale_factor
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
            if self.opt.depth_align == 'scale':
                txt_str += " Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(ratio, 0.)
                print(txt_str)
            elif self.opt.depth_align == 'scale_shift':
                txt_str += " Aligning shift and scale | t_gt: {:0.3f} | s_gt: {:0.3f} | t_pred: {:0.3f} | s_pred: {:0.3f}\n".format(
                    t_gt, s_gt, t_pred, s_pred)
                print(txt_str)
            
            error = np.array(error)
            txt_str += ("{:>11}      | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "tae", "tas")
            txt_str += "\nmean:" + ("&{: 12.3f}      " * 9).format(*(error.tolist()+[error_tae, error_tas])) + "\\\\\n\n"
            print(txt_str)
            with open(os.path.join(model_folder, 'results.txt'), 'a') as f:
                f.write(txt_str)
            
            # eval pose
            pred_poses, pred_intrinsics = [], []
            for idx in range(len(poses)-1):
                color_0, color_1 = input_colors[idx].astype(np.float32)/255., input_colors[idx+1].astype(np.float32)/255.
                color_0 = torch.from_numpy(color_0).unsqueeze(0).permute(0,3,1,2).cuda()
                color_1 = torch.from_numpy(color_1).unsqueeze(0).permute(0,3,1,2).cuda()
                all_color_aug = torch.cat([color_1, color_0], 1)
                features = [self.models["pose_encoder"](all_color_aug)]
                axisangle, translation, intermediate_feature = self.models["pose"](features)
                pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                if self.opt.learn_intrinsics:
                    cam_K = self.models['intrinsics_head'](
                            intermediate_feature, self.opt.width, self.opt.height)
                    pred_intrinsics.append(cam_K[:,:3,:3].cpu().numpy())
            pred_poses = np.concatenate(pred_poses)
            if self.opt.learn_intrinsics:
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

            sq_str = "\nsq Trajectory error: {:0.4f}, std: {:0.4f}, 95% cls: [{:0.4f}, {:0.4f}]\n".format(np.mean(ates_1), np.std(ates_1), cls_1[0], cls_1[1])
            sq_str += "sq Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res_1), np.std(res_1))
            # print(sq_str)
            if self.opt.learn_intrinsics:
                fx_mean, fx_std = np.mean(pred_intrinsics[:,0,0]) / self.opt.width, np.std(pred_intrinsics[:,0,0]) / self.opt.width
                fy_mean, fy_std = np.mean(pred_intrinsics[:,1,1]) / self.opt.height, np.std(pred_intrinsics[:,1,1]) / self.opt.height
                cx_mean, cx_std = np.mean(pred_intrinsics[:,0,2]) / self.opt.width, np.std(pred_intrinsics[:,0,2]) / self.opt.width
                cy_mean, cy_std = np.mean(pred_intrinsics[:,1,2]) / self.opt.height, np.std(pred_intrinsics[:,1,2]) / self.opt.height
                intrinsics_str = "fx: {:0.4f}, std: {:0.4f}\n".format(fx_mean, fx_std)
                intrinsics_str += "fy: {:0.4f}, std: {:0.4f}\n".format(fy_mean, fy_std)
                intrinsics_str += "cx: {:0.4f}, std: {:0.4f}\n".format(cx_mean, cx_std)
                intrinsics_str += "cy: {:0.4f}, std: {:0.4f}\n".format(cy_mean, cy_std)
                # print(intrinsics_str)
                # print(intrinsics_str)

            with open(os.path.join(save_folder, "pose_eval.txt"), "a") as f:
                f.write(sq_str)
                if self.opt.learn_intrinsics:
                    f.write(intrinsics_str + "\n")

            if self.opt.visualize_depth:
                vis_pose_dir = os.path.join(save_folder, "pose")
                os.makedirs(vis_pose_dir, exist_ok=True)
                vis_pose_sq(pred_poses, gt_local_poses, save_path=os.path.join(vis_pose_dir, f"{sequence}_{keyframe}.png"))
                
        txt_str = "\n"
        if self.opt.depth_align == 'scale':
            ratios = np.array(ratios)
            med = np.median(ratios)
            txt_str += "\n Scaling ratios | med: {:0.3f} | std: {:0.3f}\n".format(med, np.std(ratios / med))
        elif self.opt.depth_align == 'scale_shift':
            t_gts = np.array(t_gts)
            s_gts = np.array(s_gts)
            t_preds = np.array(t_preds)
            s_preds = np.array(s_preds)
            txt_str += "\n Aligning shift and scale | t_gt: {:0.3f} | s_gt: {:0.3f} | t_pred: {:0.3f} | s_pred: {:0.3f}\n".format(
                np.mean(t_gts), np.mean(s_gts), np.mean(t_preds), np.mean(s_preds))
            
        
        mean_errors = np.array(errors).mean(0)
        mean_errors_temp = np.array(errors_temp).mean(0)
        mean_errors = np.concatenate([mean_errors, mean_errors_temp])
        writer = self.writers["train"]
        for i in range(len(mean_errors)):
            writer.add_scalar(self.depth_metric_names[i], mean_errors[i], self.epoch)
        txt_str += "Epoch {:02d}".format(self.epoch)
        txt_str += "\n  " + ("{:>8} | " * 9).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "tae", "tas")
        txt_str += "\n  " + ("&{: 8.3f}  " * 9).format(*mean_errors.tolist()) + "\n"
        with open(os.path.join(model_folder, "results.txt"), 'a') as f:
            f.write(txt_str)
        print(txt_str)
        
        self.set_train()
        
        return mean_errors[2], mean_errors[4]
    def process_batch_0(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        outputs.update(self.predict_poses_0(inputs))
        losses = self.compute_losses_0(inputs, outputs)

        return outputs, losses

    def predict_poses_0(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":

                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:
                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)],
                                                                                       outputs[(
                                                                                       "position", "high", scale, f_i)])

                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width],
                            mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)], _ = self.get_occu_mask_backward(
                            outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(
                            outputs[("position", "high", scale, f_i)],
                            outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:
                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear",
                            align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[
                            ("occu_mask_backward", 0, f_i)].detach() + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0,
                                                                       max=1.0)
        return outputs

    def compute_losses_0(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            loss_smooth_registration = 0
            loss_registration = 0

            color = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                loss_smooth_registration += (get_smooth_loss(outputs[("position", scale, frame_id)], color))
                loss_registration += (
                    self.compute_reprojection_loss(outputs[("registration", scale, frame_id)], outputs[("refined", scale, frame_id)].detach()) * occu_mask_backward).sum() / occu_mask_backward.sum()

            loss += loss_registration / 2.0
            loss += self.opt.position_smoothness * (loss_smooth_registration / 2.0) / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0].unflatten(0, (self.opt.batch_size, self.opt.T)))

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, disps):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
                
            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    
                    inputs_all = [pose_feats[f_i], pose_feats[0]]
                    inputs_all_reverse = [pose_feats[0], pose_feats[f_i]]

                    # position
                    position_inputs = self.models["position_encoder"](torch.cat(inputs_all, 1))
                    position_inputs_reverse = self.models["position_encoder"](torch.cat(inputs_all_reverse, 1))
                    outputs_0 = self.models["position"](position_inputs)
                    outputs_1 = self.models["position"](position_inputs_reverse)

                    for scale in self.opt.scales:

                        outputs[("position", scale, f_i)] = outputs_0[("position", scale)]
                        outputs[("position", "high", scale, f_i)] = F.interpolate(
                            outputs[("position", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("registration", scale, f_i)] = self.spatial_transform(inputs[("color", f_i, 0)], outputs[("position", "high", scale, f_i)])
                    
                        outputs[("position_reverse", scale, f_i)] = outputs_1[("position", scale)]
                        outputs[("position_reverse", "high", scale, f_i)] = F.interpolate(
                            outputs[("position_reverse", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("occu_mask_backward", scale, f_i)],  outputs[("occu_map_backward", scale, f_i)]= self.get_occu_mask_backward(outputs[("position_reverse", "high", scale, f_i)])
                        outputs[("occu_map_bidirection", scale, f_i)] = self.get_occu_mask_bidirection(outputs[("position", "high", scale, f_i)],
                                                                                                          outputs[("position_reverse", "high", scale, f_i)])

                    # transform
                    transform_input = [outputs[("registration", 0, f_i)], inputs[("color", 0, 0)]]
                    transform_inputs = self.models["transform_encoder"](torch.cat(transform_input, 1))
                    outputs_2 = self.models["transform"](transform_inputs)

                    for scale in self.opt.scales:

                        outputs[("transform", scale, f_i)] = outputs_2[("transform", scale)]
                        outputs[("transform", "high", scale, f_i)] = F.interpolate(
                            outputs[("transform", scale, f_i)], [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
                        outputs[("refined", scale, f_i)] = (outputs[("transform", "high", scale, f_i)] * outputs[("occu_mask_backward", 0, f_i)].detach()  + inputs[("color", 0, 0)])
                        outputs[("refined", scale, f_i)] = torch.clamp(outputs[("refined", scale, f_i)], min=0.0, max=1.0)
                        # outputs[("grad_refined", scale, f_i)] = get_gradmap(outputs[("refined", scale, f_i)])
                                                                                            

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation, intermediate_feature = self.models["pose"](pose_inputs)

                    if self.opt.learn_intrinsics:
                        cam_K = self.models['intrinsics_head'](
                        intermediate_feature, self.opt.width, self.opt.height)
                        inv_K = torch.inverse(cam_K)
                        outputs[('K', 0)] = cam_K
                        outputs[('inv_K', 0)] = inv_K
                    
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])
                    
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            if not self.opt.learn_intrinsics:
                cam_K = inputs[("K", source_scale)]
                inv_K = inputs[("inv_K", source_scale)]
            else:
                cam_K = outputs[('K', source_scale)]
                inv_K = outputs[('inv_K', source_scale)]
                # if self.step % (self.opt.log_frequency*5) == 0:
                #     print("predicted K:", cam_K[0])
                #     print("true K:", inputs[("K", 0)][0] )
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inv_K) # [B, 4, H*W]
                pix_coords, src_depths = self.project_3d[source_scale](
                    cam_points, cam_K, T) # [B, H, W, 2]

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, cam_K, T)
                
                # reproject depth
                if frame_id == 1:
                    tgt_depth = depth[1:]
                    src_coords = pix_coords[:-1]
                    src_depth = src_depths[:-1].reshape(*tgt_depth.shape) # [(B-1), 1, H, W]
                elif frame_id == -1:
                    tgt_depth = depth[:-1]
                    src_coords = pix_coords[1:]
                    src_depth = src_depths[1:].reshape(*tgt_depth.shape) # [(B-1), 1, H, W]
                sampled_depth = F.grid_sample(
                    tgt_depth,
                    src_coords,
                    padding_mode="zeros",
                    align_corners=True) # [B-1, 1, H, W]
                proj_mask = sampled_depth > 1e-3
                outputs[("reproj_depth_error", scale, frame_id)] = torch.abs(src_depth-sampled_depth)[proj_mask].mean()

                # optical flow
                if frame_id == 1:
                    origin_depth = depth[:-1]
                    flow_map = outputs[("position", "high", scale, frame_id)][:-1]
                    forward_depth = depth[1:]
                elif frame_id == -1:
                    origin_depth = depth[1:]
                    flow_map = outputs[("position", "high", scale, frame_id)][1:]
                    forward_depth = depth[:-1]
                warp_depth = self.spatial_transform(origin_depth, flow_map, padding='zeros')
                warp_mask = warp_depth > 1e-3
                outputs[("flow_depth_error", scale, frame_id)] = torch.abs(warp_depth-forward_depth)[warp_mask].mean()

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            
            loss = 0
            loss_reprojection = 0
            loss_transform = 0
            loss_cvt = 0
            loss_depth_reproj = 0
            loss_depth_flow = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            if disp.shape[-2:] != color.shape[-2:]:
                disp = F.interpolate(
                        disp, [color.shape[-2], color.shape[-1]], mode="bilinear", align_corners=True)

            for frame_id in self.opt.frame_ids[1:]:
                
                occu_mask_backward = outputs[("occu_mask_backward", 0, frame_id)].detach()
                
                loss_reprojection += (
                    self.compute_reprojection_loss(outputs[("color", frame_id, scale)], outputs[("refined", scale, frame_id)]) * occu_mask_backward).sum() / occu_mask_backward.sum()  
                loss_transform += (
                    torch.abs(outputs[("refined", scale, frame_id)] - outputs[("registration", 0, frame_id)].detach()).mean(1, True) * occu_mask_backward).sum() / occu_mask_backward.sum()
                loss_cvt += get_smooth_bright(
                    outputs[("transform", "high", scale, frame_id)], inputs[("color", 0, 0)], outputs[("registration", scale, frame_id)].detach(), occu_mask_backward)
                loss_depth_reproj += outputs[("reproj_depth_error", scale, frame_id)]
                loss_depth_flow += outputs[("flow_depth_error", scale, frame_id)]

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            temporal_weight = 1. if self.tune_temporal else 0.

            loss_reprojection = loss_reprojection / 2.0
            loss_transform = self.opt.transform_constraint * loss_transform / 2.0
            loss_cvt = self.opt.transform_smoothness * loss_cvt / 2.0
            loss_smooth = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss_depth_reproj = temporal_weight * self.opt.depth_reproj * loss_depth_reproj / 2.0
            loss_depth_flow = temporal_weight * self.opt.depth_flow * loss_depth_flow / 2.0
            loss += loss_reprojection + loss_transform + loss_cvt + loss_smooth + loss_depth_reproj + loss_depth_flow

            total_loss += loss
            losses["loss/{}".format(scale)] = loss.item()
            losses["loss/loss_reprojection/{}".format(scale)] = loss_reprojection.item()
            losses["loss/loss_transform/{}".format(scale)] = loss_transform.item()
            losses["loss/loss_cvt/{}".format(scale)] = loss_cvt.item()
            losses["loss/loss_smooth/{}".format(scale)] = loss_smooth.item()
            losses["loss/loss_depth_reproj/{}".format(scale)] = loss_depth_reproj.item()
            losses["loss/loss_depth_flow/{}".format(scale)] = loss_depth_flow.item()

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
    
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            # flatten Batch_size and Time dims
            if self.opt.T > -1:
                for key in inputs.keys():
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].flatten(0, 1)
            outputs, losses = self.process_batch_val(inputs)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch_val(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        outputs = self.models["depth_model"](inputs["color_aug", 0, 0].unflatten(0, (self.opt.batch_size, self.opt.T)))

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, outputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses_val(inputs, outputs)

        return outputs, losses

    def compute_losses_val(self, inputs, outputs):
        """Compute the reprojection, perception_loss and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:

            loss = 0
            registration_losses = []

            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                registration_losses.append(
                    ncc_loss(outputs[("registration", scale, frame_id)].mean(1, True), target.mean(1, True)))

            registration_losses = torch.cat(registration_losses, 1)
            registration_losses, idxs_registration = torch.min(registration_losses, dim=1)

            loss += registration_losses.mean()
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = -1 * total_loss

        return losses

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids[1:]:

                    writer.add_image(
                        "brightness_{}_{}/{}".format(frame_id, s, j),
                        outputs[("transform", "high", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "registration_{}_{}/{}".format(frame_id, s, j),
                        outputs[("registration", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "refined_{}_{}/{}".format(frame_id, s, j),
                        outputs[("refined", s, frame_id)][j].data, self.step)
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "occu_mask_backward_{}_{}/{}".format(frame_id, s, j),
                            outputs[("occu_mask_backward", s, frame_id)][j].data, self.step)
            
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, mode='epoch'):
        """Save model weights to disk
        """
        if mode == 'epoch':
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        elif mode == 'last':
            save_folder = os.path.join(self.log_path, "models", "weights_last")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'depth_model':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
            # print("Loading Adam weights")
            # optimizer_dict = torch.load(optimizer_load_path)
            # self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        print("Adam is randomly initialized")

