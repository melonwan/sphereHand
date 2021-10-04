from __future__ import absolute_import, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.hourglass import create_hourglass_network
from mesh.multiview_utility import (
    MultiviewConsistencyLoss, 
    MutualProjectionLoss,
    WeightedMultiviewConsistencyLoss
    )
from network.util_modules import (
    ResizeCropImage,
    RecoverXYZCoordinateFromHeatmap, 
    HeatmapVariance,
    PosePriorLoss,
    TemporalSmoothnessLoss)
from network.pose_vae import(
    PoseVae
)
from mesh.render import (
    CollisionLoss,
    BoneLengthLoss
)

class HeatmapEstimationNetwork(nn.Module):
    def __init__(self, heatmap_size, depth_scale, num_joints, num_stacks, real_aug: bool=True):
        super(HeatmapEstimationNetwork, self).__init__()
        self.num_joints = num_joints
        self.hg = create_hourglass_network(num_joints * 2, num_stacks)
        self.xyz_recover = RecoverXYZCoordinateFromHeatmap(heatmap_size, 
                                                           heatmap_size,
                                                           depth_scale)
        self.resize_dm = ResizeCropImage() if real_aug else None

    def _foward_real(self, dms):
        num_real = dms.shape[0]
        num_view = dms.shape[1]
        dms = dms.view(num_real*num_view, dms.shape[2], dms.shape[3])
        if self.resize_dm is not None and self.training:
            if torch.rand(1).item() < 0.5:
                rnd_u_scale = torch.ones(num_real*num_view).cuda()
                rnd_v_scale = torch.ones(num_real*num_view).cuda()
            else:
                rnd_scale = torch.rand(num_real*num_view).cuda() *0.2 + 0.75
                rnd_u_scale = (rnd_scale + torch.rand_like(rnd_scale)*0.1 - 0.05)
                rnd_v_scale = (rnd_scale + torch.rand_like(rnd_scale)*0.1 - 0.05)
                dms = self.resize_dm(dms, rnd_u_scale, rnd_v_scale)
        else:
            rnd_u_scale = torch.ones(num_real*num_view).cuda()
            rnd_v_scale = torch.ones(num_real*num_view).cuda()
        output, _ = self.hg(dms)

        result = {}
        real_uv_hms = [o[:, :self.num_joints] for o in output]
        real_d_hms = [o[:, self.num_joints:] for o in output]
        real_xyz = [self.xyz_recover(uv_hm, d_hm, True) for uv_hm, d_hm in zip(real_uv_hms, real_d_hms)]
        for xyz in real_xyz:
            xyz[:,:,0] /= rnd_u_scale.view(-1, 1)
            xyz[:,:,1] /= rnd_v_scale.view(-1, 1)

        result['real_uv_hms'] = [h.reshape(num_real, 
                                          num_view, 
                                          self.num_joints, 
                                          h.shape[-2],
                                          h.shape[-1]) for h in real_uv_hms]
        result['real_d_hms'] = [h.reshape(num_real, 
                                          num_view, 
                                          self.num_joints, 
                                          h.shape[-2],
                                          h.shape[-1]) for h in real_d_hms]
        result['real_xyz'] = [p.reshape(num_real, num_view, self.num_joints, 3) for p in real_xyz]
        return result
    
    def _foward_synthetic(self, dms):
        output, _ = self.hg(dms)
        result = {}
        result['synt_uv_hms'] = [o[:, :self.num_joints] for o in output]
        result['synt_d_hms'] = [o[:, self.num_joints:] for o in output]
        result['synt_xyz'] = [self.xyz_recover(uv_hm, d_hm) for uv_hm, d_hm in zip(result['synt_uv_hms'], result['synt_d_hms'])]
        return result
    
    def forward(self, real_dms: torch.tensor=None, synt_dms: torch.tensor=None):
        if synt_dms is None:
            return self._foward_real(real_dms)
        if real_dms is None:
            return self._foward_synthetic(synt_dms)
        
        num_sync = synt_dms.shape[0]
        num_real = real_dms.shape[0]
        num_view = real_dms.shape[1]
        real_dms = real_dms.view(num_real*num_view, real_dms.shape[2], real_dms.shape[3])
        if self.resize_dm is not None and self.training:
            if torch.rand(1).item() < 0.5:
                rnd_u_scale = torch.ones(num_real*num_view).cuda()
                rnd_v_scale = torch.ones(num_real*num_view).cuda()
            else:
                rnd_scale = torch.rand(num_real*num_view).cuda() *0.2 + 0.75
                rnd_u_scale = (rnd_scale + torch.rand_like(rnd_scale)*0.1 - 0.05)
                rnd_v_scale = (rnd_scale + torch.rand_like(rnd_scale)*0.1 - 0.05)
                real_dms = self.resize_dm(real_dms, rnd_u_scale, rnd_v_scale)
        else:
            rnd_u_scale = torch.ones(num_real*num_view).cuda()
            rnd_v_scale = torch.ones(num_real*num_view).cuda()

        combined_dms = torch.cat([synt_dms, real_dms], dim=0)
        combined_output, combined_latent = self.hg(combined_dms)

        result = {}
        result['synt_uv_hms'] =\
            [o[:num_sync, :self.num_joints, :, :] for o in combined_output]
        result['synt_d_hms'] =\
            [o[:num_sync, self.num_joints:, :, :] for o in combined_output]
        result['synt_xyz'] =\
            [self.xyz_recover(uv_hm, d_hm) for uv_hm, d_hm in zip(result['synt_uv_hms'], result['synt_d_hms'])]

        real_uv_hms =\
         [o[num_sync:, :self.num_joints, :, :] for o in combined_output]
        real_d_hms =\
         [o[num_sync:, self.num_joints:,:,:] for o in combined_output]
        real_xyz =\
         [self.xyz_recover(uv_hm, d_hm, True) for uv_hm, d_hm in zip(real_uv_hms, real_d_hms)]
        for xyz in real_xyz:
            xyz[:,:,0] /= rnd_u_scale.view(-1, 1)
            xyz[:,:,1] /= rnd_v_scale.view(-1, 1)

        result['real_uv_hms'] = [h.reshape(num_real, 
                                           num_view, 
                                           self.num_joints,
                                           h.shape[-2], 
                                           h.shape[-1]) for h in real_uv_hms]
        result['real_d_hms'] = [h.reshape(num_real, 
                                          num_view, 
                                          self.num_joints,
                                          h.shape[-2], 
                                          h.shape[-1]) for h in real_d_hms]
        result['real_xyz'] = [p.reshape(num_real, num_view, self.num_joints, 3) for p in real_xyz]
        if self.resize_dm is not None:
            result['real_resized_dms'] = real_dms

        result['batch_synt_fea'] = [l[:num_sync] for l in combined_latent]
        result['batch_real_fea'] = [l[num_sync:] for l in combined_latent]
        return result


class MultiTaskLoss(nn.Module):
    def __init__(self, 
                 synthesized_loss: bool,
                 mv_projection_loss: bool,
                 mv_consistency_loss: bool,
                 temporal_smooth_loss: bool,
                 prior_loss: bool,
                 collision_loss: bool,
                 bone_length_loss: bool,
                 constant: any,
                 image_size: int = 64,
                 heatmap_size: int = 16):
        super(MultiTaskLoss, self).__init__()
        self.synthesized_loss = nn.MSELoss() if synthesized_loss else None
        self.mv_projection_loss = MutualProjectionLoss(image_size, constant.mesh) if mv_projection_loss else None
        self.mv_consistency_loss = MultiviewConsistencyLoss() if mv_consistency_loss else None
        self.temporal_smooth_loss = TemporalSmoothnessLoss() if temporal_smooth_loss else None
        self.prior_loss = PoseVae(41*3, 32, 'mesh/model/pose_vae.pth') if prior_loss else None
        self.collision_criterion = CollisionLoss() if collision_loss else None
        self.heatmap_variance = HeatmapVariance(heatmap_size, heatmap_size)
        self.bone_length_criterion = BoneLengthLoss() if bone_length_loss else None
        self.domain_loss = nn.MSELoss()
        self.heatmap_size = heatmap_size

        self.weights = {}
        self.weights['synt_hm'] = 1e3
        self.weights['synt_pt'] = 1e-1
        self.weights['mv_consistency'] = 1e-3
        self.weights['mv_projection'] = 1
        self.weights['temporal_smooth'] = 1.0
        self.weights['prior'] = 1e-2
        self.weights['hm_mean'] = 1e-2
        self.weights['domain'] = 0.0
        self.weights['collision'] = 1.0
        self.weights['bone_length'] = 1.0

    def forward(self, 
                result: dict, 
                synt_target: dict=None, 
                real_target: dict=None):
        loss_terms = {}
        if self.synthesized_loss is not None and synt_target is not None:
            loss_terms['synt_uv'] = 0
            for est_hm in result['synt_uv_hms']:
                loss_terms['synt_uv'] +=\
                    self.weights['synt_hm'] * self.synthesized_loss(est_hm, 
                                                           synt_target['uv_hms'])
            loss_terms['synt_d'] = 0
            target_z_pts = synt_target['xyz_pts'][:,:,2]
            for xyz in result['synt_xyz']:
                z_pts = xyz[:,:,2]
                loss_terms['synt_d'] +=\
                  self.weights['synt_pt'] * self.synthesized_loss(z_pts, target_z_pts)

        projected_dms = []
        if real_target is None:
            is_mv = False
        else:
            is_mv = real_target['is_mv'] if 'is_mv' in real_target else True
        # is_mv = False
        
        if self.mv_projection_loss is not None and real_target is not None:
            loss_terms['mv_projection'] = 0
            for xyz in result['real_xyz']:
                curr_stack_loss, dm = \
                  self.mv_projection_loss(
                    real_target['camera_poses'], 
                    real_target['inv_camera_poses'],
                    xyz, 
                    real_target['real_dms'],
                    is_mv)
                curr_stack_loss = curr_stack_loss * self.weights['mv_projection']
                loss_terms['mv_projection'] += curr_stack_loss
                projected_dms.append(dm)
        
        if self.mv_consistency_loss is not None and real_target is not None:
            loss_terms['mv_consistency'] = 0
            for xyz, uv_hm in zip(result['real_xyz'], result['real_uv_hms']):
                hm_weight = None
                w = self.weights['mv_consistency'] if is_mv else 0
                loss_terms['mv_consistency'] +=\
                    w * self.mv_consistency_loss(
                     real_target['camera_poses'], xyz, hm_weight)
        
        if real_target is not None:
            loss_terms['uv_hm_mean'] = 0
            for est_hm in result['real_uv_hms']:
                est_hm = est_hm.reshape(est_hm.shape[0]*est_hm.shape[1], est_hm.shape[2], self.heatmap_size, self.heatmap_size)
                loss_terms['uv_hm_mean'] += self.weights['hm_mean'] * self.synthesized_loss(est_hm, torch.zeros_like(est_hm))
        
        if self.prior_loss is not None:
            loss_terms['pose_prior'] = 0
            for xyz in result['real_xyz']:
                loss_terms['pose_prior'] += self.weights['prior'] * self.prior_loss.prior_loss(xyz / 100.0)

        if self.temporal_smooth_loss is not None:
            loss_terms['temporal_smooth'] = 0
            for xyz in result['real_xyz']:
                loss_terms['temporal_smooth'] += self.weights['temporal_smooth'] * self.temporal_smooth_loss(xyz)
        
        if self.collision_criterion is not None:
            loss_terms['collision'] = 0
            for xyz in result['real_xyz']:
                loss_terms['collision'] += self.weights['collision'] * self.collision_criterion(xyz)
        
        if self.bone_length_criterion is not None:
            loss_terms['bone_length'] = 0
            for xyz in result['real_xyz']:
                loss_terms['bone_length'] += self.weights['bone_length'] * self.bone_length_criterion(xyz)

        if 'batch_synt_fea' in result and 'batch_real_fea' in result:
            loss_terms['domain_loss'] = 0
            for synt, real in zip(result['batch_synt_fea'], result['batch_real_fea']):
                synt = synt.mean(dim=0).mean(dim=-1).mean(dim=-1)
                real = real.mean(dim=0).mean(dim=-1).mean(dim=-1)
                loss_terms['domain_loss'] += self.weights['domain'] * self.domain_loss(synt, real)
        return loss_terms, projected_dms
        


def result_shape_info(result) -> str:
    if type(result) == list:
        info = [str(r.shape) for r in result]
        return ' '.join(info) + ' '

    if type(result) == dict:
        info = ''
        for k, v in result.items():
                info += '{}: {}'.format(k, result_shape_info(v))
        return info

def combine_loss(loss_terms: dict) -> torch.tensor:
    loss = 0
    for _, l in loss_terms.items():
        loss += l
    return loss