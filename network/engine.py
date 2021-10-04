from __future__ import print_function, absolute_import, division

import torch
import torch.utils.data as data
import numpy as np
import os
import random
import string
import cv2
import time
import json
from enum import Enum
from network.create_network_and_criterion import HeatmapEstimationNetwork, MultiTaskLoss
from network.util_modules import HandSynthesizer, RecoverXYZCoordinateFromHeatmap, DepthResample
from dataset.joint_angle import JointAngleDataset
from dataset.nyu_dataset import create_nyu_dataset
from network.constants import Constant
from network.util_vis import vis_result, vis_depthmap
from network.utils_metric import average_joint_error
from network.pose_denoiser import PoseDenoiser
from mesh.multiview_utility import FuseMvPose
constant = Constant()


class Mode(Enum):
    Train = 1
    Eval = 2


class RunningAverage():
    def __init__(self):
        self.num = 0
        self.sum = None

    def append(self, data: dict):
        if self.sum is None:
            self.sum = data
        else:
            for k, v in data.items():
                self.sum[k] += float(v)
        self.num += 1

    def __str__(self) -> str:
        info = ''
        if self.sum is None:
            return info
        for k, v in self.sum.items():
            info += '{}: {:.4f} '.format(k, float(v / self.num))
        return info


class Engine():
    def __init__(self, opts):
        self.network = HeatmapEstimationNetwork(
            constant.heatmap_size, constant.depth_scale, constant.num_joint, opts.num_stacks).cuda()
        self.criterion = MultiTaskLoss(opts.synthesize,
                                       opts.mv_projection,
                                       opts.mv_consistency,
                                       opts.temporal,
                                       opts.prior,
                                       opts.collision,
                                       opts.bone_length,
                                       constant,
                                       image_size=constant.depthmap_size).cuda()
        self.hand_synthesizer = HandSynthesizer(
            constant.mesh,
            image_size=constant.depthmap_size,
            heatmap_size=constant.heatmap_size,
            uv_hm_scale=constant.uv_hm_scale,
            depth_scale=constant.depth_scale).cuda()
        self.pose_denoiser = PoseDenoiser(
            model_path='mesh/model/pose_denoiser.pth').cuda()
        self.pose_denoiser.eval()
        self.fuse_mv_pose = FuseMvPose().cuda()
        self.coor_from_hms = RecoverXYZCoordinateFromHeatmap(
            constant.heatmap_size,
            constant.heatmap_size,
            constant.depth_scale).cuda()
        self.depth_segmentation = None
        self.num_stacks = opts.num_stacks
        self.temporal_smooth = opts.temporal

        if opts.depth_resample == 0:
            self.depth_sampler = None
        else:
            self.depth_sampler = DepthResample(
                0.95, opts.depth_resample).cuda()

        self.model_dir = opts.model_dir
        self.mode = Mode.Train if opts.mode == 'Train' else Mode.Eval
        if self.mode is Mode.Eval:
            assert opts.initial_model is not None

        self.epoch = opts.epoch
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=opts.lr,
                                          weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.epoch//3, gamma=0.1)
        self.starting_epoch = 0

        def rand_model_name() -> str:
            return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))

        if opts.restore_from_model is not None:
            self.model_name = opts.restore_from_model
            self.model_path = os.path.join(self.model_dir, self.model_name)
            self.load_model(opts.restore_from_epoch)
        else:
            self.model_name = opts.tag + rand_model_name()
            self.model_path = os.path.join(self.model_dir, self.model_name)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        print('[engine] the model will be saved to: {}'.format(self.model_path))

        with open(os.path.join(self.model_path, 'loss_weights.txt'), 'w') as f:
            json.dump(self.criterion.weights, f)

        if opts.initial_model is not None:
            self.load_model(opts.initial_model)
            self.fine_tune = True
        else:
            self.fine_tune = False

        self.log_file = os.path.join(self.model_path, 'log.txt')

        self.image_dir = os.path.join(self.model_path, 'images')
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.real_train_dataset = create_nyu_dataset([
            os.path.join(opts.dataset_dir, 'train')])
        self.real_eval_dataset = create_nyu_dataset(
            os.path.join(opts.dataset_dir, 'test'))

        self.synt_dataset = JointAngleDataset()
        self.with_synt = opts.synthesize
        self.with_real = any(
            [opts.mv_projection, opts.mv_consistency, opts.temporal, opts.prior, opts.collision, opts.bone_length])

    def curr_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def sum_loss_terms(self, loss_terms: dict) -> torch.tensor:
        loss = 0
        for _, l in loss_terms.items():
            loss += l
        return loss

    def _epoch_with_real(self, mode: Mode, epoch: int, save_result: bool = False):
        if mode == Mode.Train:
            self.network.train()
            real_dataset = self.real_train_dataset
        else:
            self.network.eval()
            real_dataset = self.real_eval_dataset

        real_loader = data.DataLoader(real_dataset,
                                      batch_size=8, shuffle=not self.temporal_smooth if mode is Mode.Train else False, num_workers=2)

        running_loss_average = RunningAverage()
        running_metric_average = RunningAverage()
        t_prev = time.time()

        epoch_gt_joints = []
        epoch_est_joints = []
        epoch_rendered_image = []

        for curr_it, (real_dms, gt_joints, camera_poses, inv_camera_poses) in enumerate(real_loader):
            real_dms = real_dms.cuda()
            gt_joints = gt_joints.cuda()
            orig_real_dms = real_dms.clone()
            real_dms = real_dms * constant.depth_scale
            camera_poses = camera_poses.cuda()
            inv_camera_poses = inv_camera_poses.cuda()
            num_views = orig_real_dms.shape[1]
            if self.depth_sampler is not None:
                real_dms = self.depth_sampler(real_dms.view(-1, constant.depthmap_size, constant.depthmap_size)
                                              ).view(-1, num_views, constant.depthmap_size, constant.depthmap_size)
            if self.depth_segmentation is not None:
                orig_real_dms = self.depth_segmentation(
                    orig_real_dms, gt_joints)

            self.optimizer.zero_grad()
            result = self.network(real_dms=real_dms)

            if curr_it % 100 == 0:
                info = 'time: {:.2f}s'.format(
                    time.time() - t_prev)
                t_prev = time.time()
                print(info)

            real_target = {}
            real_target['real_dms'] = orig_real_dms
            real_target['camera_poses'] = camera_poses
            real_target['inv_camera_poses'] = inv_camera_poses
            loss_terms, ball_dms = self.criterion(
                result, real_target=real_target)
            loss = self.sum_loss_terms(loss_terms)
            running_loss_average.append(loss_terms)
            metrics = {}
            if mode is Mode.Eval:
                gt_joints = gt_joints[:, 0].unsqueeze(dim=1)
                est_joints = self.pose_denoiser(
                    result['real_xyz'][-1][:, 0]).unsqueeze(dim=1)
                metrics['avg_joint_error'] = average_joint_error(
                    gt_joints=gt_joints, est_joints=est_joints)
            else:
                est_joints = result['real_xyz'][-1]
                metrics['avg_joint_error'] = average_joint_error(
                    gt_joints=gt_joints, est_joints=est_joints)
            epoch_gt_joints.append(gt_joints.detach().cpu().numpy())
            epoch_est_joints.append(est_joints.detach().cpu().numpy())
            epoch_rendered_image.append(ball_dms[-1].detach().cpu().numpy())

            running_metric_average.append(metrics)
            if mode == Mode.Train:
                loss.backward()
                self.optimizer.step()

            if curr_it % 100 == 0:
                info = '[{}-{}]: metric: {}, loss: {}, lr: {}, time: {:.2f}s'.format(
                    epoch, curr_it, running_metric_average, running_loss_average, self.curr_lr(), time.time() - t_prev)
                print(info)
                with open(self.log_file, 'a') as f:
                    f.write(info + '\n')
                t_prev = time.time()

            if curr_it % 100 == 0:
                real_dms = real_dms.view(-1, constant.depthmap_size,
                                         constant.depthmap_size)
                real_uv_hms = result['real_uv_hms'][-1]
                real_uv_hms = real_uv_hms.view(
                    -1, constant.num_joint, constant.heatmap_size, constant.heatmap_size)
                real_xyz = result['real_xyz'][-1]
                real_xyz = real_xyz.view(-1, constant.num_joint, 3)
                img = vis_result(real_dms, real_uv_hms,
                                 real_xyz, constant.synt_key_points)
                img_path = os.path.join(
                    self.image_dir, '%s_%d_%d.jpg' % (mode.name, epoch, curr_it))

                if len(ball_dms) != 0:
                    ball_dm = ball_dms[-1]
                    ball_dm = ball_dm * constant.depth_scale
                    ball_dm = ball_dm.view(ball_dm.shape[0]*ball_dm.shape[1],
                                           ball_dm.shape[2],
                                           ball_dm.shape[3],
                                           ball_dm.shape[4])
                    ball_dm = ball_dm.detach().cpu().numpy()
                    images = []
                    for mv_dm in ball_dm:
                        mv_images = []
                        for dm in mv_dm:
                            mv_images.append(vis_depthmap(dm))
                        images.append(np.hstack(mv_images))
                    ball_img = np.vstack(images)
                    ball_img = cv2.resize(
                        ball_img, (img.shape[1], img.shape[0]))
                    img = np.hstack([img, ball_img])
                cv2.imwrite(img_path, img)

        print('[epoch: {}]: metric: {}, loss: {}, lr: {}'.format(
            epoch, running_metric_average, running_loss_average, self.curr_lr()))

    def _epoch_with_synt(self, mode: Mode, epoch: int, save_result: bool = False):
        if mode == Mode.Train:
            self.network.train()
        else:
            self.network.eval()

        pose_para_loader = data.DataLoader(
            self.synt_dataset, batch_size=128 // self.num_stacks, shuffle=True, num_workers=1)
        pose_para_it = iter(pose_para_loader)
        running_loss_average = RunningAverage()
        t_prev = time.time()

        epoch_gt_joints = []
        epoch_est_joints = []

        for curr_it in range(1000 * self.num_stacks):
            self.optimizer.zero_grad()
            pose_parameter = next(pose_para_it).cuda()
            dms, uv_hms, d_hms, synt_xyz_pts =\
                self.hand_synthesizer(pose_parameter)
            result = self.network(synt_dms=dms)
            synt_target = {'uv_hms': uv_hms,
                           'd_hms': d_hms,
                           'xyz_pts': synt_xyz_pts}
            loss_terms, _ = self.criterion(result, synt_target)
            loss = self.sum_loss_terms(loss_terms)
            running_loss_average.append(loss_terms)

            epoch_gt_joints.append(synt_xyz_pts.detach().cpu().numpy())
            epoch_est_joints.append(
                result['synt_xyz'][-1].detach().cpu().numpy())

            if mode == Mode.Train:
                loss.backward()
                self.optimizer.step()
            if curr_it % 500 == 0:
                est_img = vis_result(
                    dms, result['synt_uv_hms'][-1], result['synt_xyz'][-1], constant.synt_key_points)
                gt_xyz = self.coor_from_hms(uv_hms, d_hms)
                gt_img = vis_result(dms, uv_hms, gt_xyz,
                                    constant.synt_key_points)
                img = np.hstack([gt_img, est_img])
                img_path = os.path.join(
                    self.image_dir, '%s_%d_%d.jpg' % (mode.name, epoch, curr_it))
                cv2.imwrite(img_path, img)
            if curr_it % 100 == 0:
                info = '[{}-{}]: loss: {}, lr: {}, time: {:.2f}s'.format(
                    epoch, curr_it, running_loss_average, self.curr_lr(), time.time() - t_prev)
                print(info)
                with open(self.log_file, 'a') as f:
                    f.write(info + '\n')
                t_prev = time.time()

    def _epoch_with_both(self, mode: Mode, epoch: int):
        if mode == Mode.Train:
            self.network.train()
            real_dataset = self.real_train_dataset
        else:
            self.network.eval()
            real_dataset = self.real_eval_dataset

        real_loader = data.DataLoader(real_dataset,
                                      batch_size=25, shuffle=not self.temporal_smooth if mode is Mode.Train else False, num_workers=2)
        pose_para_loader = data.DataLoader(
            self.synt_dataset, batch_size=48, shuffle=True, num_workers=1)
        pose_para_it = iter(pose_para_loader)
        running_loss_average = RunningAverage()
        running_metric_average = RunningAverage()
        t_prev = time.time()
        for curr_it, (real_dms, gt_joints, camera_poses, inv_camera_poses) in enumerate(real_loader):
            real_dms = real_dms.cuda()
            orig_real_dms = real_dms.clone()
            real_dms = real_dms * constant.depth_scale
            camera_poses = camera_poses.cuda()
            inv_camera_poses = inv_camera_poses.cuda()
            gt_joints = gt_joints.cuda()
            num_views = orig_real_dms.shape[1]
            if self.depth_sampler is not None:
                real_dms = self.depth_sampler(real_dms.view(-1, constant.depthmap_size, constant.depthmap_size)
                                              ).view(-1, num_views, constant.depthmap_size, constant.depthmap_size)
            if self.depth_segmentation is not None:
                orig_real_dms = self.depth_segmentation(
                    orig_real_dms, gt_joints)

            pose_parameter = next(pose_para_it).cuda()
            synt_dms, uv_hms, d_hms, synt_xyz_pts =\
                self.hand_synthesizer(pose_parameter)
            if self.depth_sampler is not None:
                synt_dms = self.depth_sampler(synt_dms).squeeze()

            self.optimizer.zero_grad()
            result = self.network(synt_dms=synt_dms, real_dms=real_dms)
            real_target = {
                'real_dms': orig_real_dms,
                'camera_poses': camera_poses,
                'inv_camera_poses': inv_camera_poses,
                'is_mv': curr_it < 1500}
            synt_target = {
                'uv_hms': uv_hms,
                'd_hms': d_hms,
                'xyz_pts': synt_xyz_pts}
            loss_terms, ball_dms = self.criterion(
                result, real_target=real_target, synt_target=synt_target)
            loss = self.sum_loss_terms(loss_terms)
            running_loss_average.append(loss_terms)
            metrics = {}
            metrics['avg_joint_error'] = average_joint_error(
                gt_joints=gt_joints, est_joints=result['real_xyz'][-1])
            running_metric_average.append(metrics)
            if mode == Mode.Train:
                loss.backward()
                self.optimizer.step()

            if curr_it % 100 == 0:
                info = '[{}-{}]: metric: {}, loss: {}, lr: {}, time: {:.2f}s'.format(
                    epoch, curr_it, running_metric_average, running_loss_average, self.curr_lr(), time.time() - t_prev)
                print(info)
                with open(self.log_file, 'a') as f:
                    f.write(info + '\n')
                t_prev = time.time()

            if curr_it % 400 == 0:
                if 'real_resized_dms' in result:
                    resized_real_dms = result['real_resized_dms'].view(
                        -1, constant.depthmap_size, constant.depthmap_size)
                else:
                    resized_real_dms = None
                real_dms = real_dms.view(-1, constant.depthmap_size,
                                         constant.depthmap_size)
                real_uv_hms = result['real_uv_hms'][-1]
                real_uv_hms = real_uv_hms.view(
                    -1, constant.num_joint, constant.heatmap_size, constant.heatmap_size)
                real_xyz = result['real_xyz'][-1]
                real_xyz = real_xyz.view(-1, constant.num_joint, 3)
                real_img = vis_result(real_dms, real_uv_hms, real_xyz,
                                      constant.synt_key_points, resized_dms=resized_real_dms)

                synt_img = vis_result(
                    synt_dms, result['synt_uv_hms'][-1], result['synt_xyz'][-1], constant.synt_key_points)

                if len(ball_dms) != 0:
                    ball_dm = ball_dms[-1]
                    ball_dm = ball_dm * constant.depth_scale
                    ball_dm = ball_dm.view(ball_dm.shape[0]*ball_dm.shape[1],
                                           ball_dm.shape[2],
                                           ball_dm.shape[3],
                                           ball_dm.shape[4])
                    ball_dm = ball_dm.detach().cpu().numpy()
                    images = []
                    for mv_dm in ball_dm:
                        mv_images = []
                        for dm in mv_dm:
                            mv_images.append(vis_depthmap(dm))
                        images.append(np.hstack(mv_images))
                    ball_img = np.vstack(images)
                    ball_img = cv2.resize(
                        ball_img, (real_img.shape[1], real_img.shape[0]))
                    real_img = np.hstack([real_img, ball_img])

                    gt_xyz = self.coor_from_hms(uv_hms, d_hms)
                    gt_img = vis_result(synt_dms,
                                        result['synt_uv_hms'][-1],
                                        gt_xyz,
                                        constant.synt_key_points)
                    synt_img = np.hstack([synt_img, gt_img])

                img = np.vstack([real_img, synt_img])
                img_path = os.path.join(
                    self.image_dir, '%s_%d_%d.jpg' % (mode.name, epoch, curr_it))
                cv2.imwrite(img_path, img)
        print('[epoch: {}]: metric: {}, loss: {}, lr: {}'.format(
            epoch, running_metric_average, running_loss_average, self.curr_lr()))

    def save_model(self, epoch):
        pth_path = os.path.join(self.model_path, 'model_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, pth_path)

    def load_model(self, epoch: any):
        if type(epoch) is int:
            pth_path = os.path.join(
                self.model_path, 'model_{}.pth'.format(epoch))
        elif type(epoch) is str:
            pth_path = epoch
        else:
            raise ValueError
        check_point = torch.load(pth_path)
        self.network.load_state_dict(check_point['network_state_dict'])
        if type(epoch) is int:
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
            self.starting_epoch = check_point['epoch']
            for _ in range(self.starting_epoch):
                self.scheduler.step()

    def train(self):
        for epoch in range(self.starting_epoch, self.epoch):
            if self.with_real and self.with_synt:
                self._epoch_with_both(Mode.Train, epoch)
            elif self.with_synt:
                self._epoch_with_synt(Mode.Train, epoch)
            elif self.with_real:
                self._epoch_with_real(Mode.Train, epoch)

            self.scheduler.step()
            self.save_model(-1)
            if epoch % 1 == 0 or epoch == self.epoch-1:
                self.save_model(epoch)

    def eval(self):
        self._epoch_with_real(Mode.Eval, 0, save_result=True)
