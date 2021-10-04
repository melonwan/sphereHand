from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mesh.render import BallRender, DataToModelLoss
from network.util_modules import HeatmapVariance

class MutualTransformation(nn.Module):
    def __self__(self):
        super(MutualTransformation, self).__init__()
    
    def forward(self, trans_mats: torch.tensor, inv_trans_mats: torch.tensor):
        assert trans_mats.ndimension() == 4
        num_views = trans_mats.shape[1]

        mutual_trans = []
        # caucluate transformation from view_i to view_j
        for view_i in range(num_views):
            i_to_rest_trans = []
            for view_j in range(num_views):
                i_to_j_trans = torch.matmul(inv_trans_mats[:,view_j],
                                       trans_mats[:,view_i])
                i_to_j_trans = i_to_j_trans.unsqueeze(dim=1)
                i_to_rest_trans.append(i_to_j_trans)
            i_to_rest_trans = torch.cat(i_to_rest_trans, dim=1)
            i_to_rest_trans = i_to_rest_trans.unsqueeze(dim=1)
            mutual_trans.append(i_to_rest_trans)
        mutual_trans = torch.cat(mutual_trans, dim=1)
        return mutual_trans

class MutualProjection(nn.Module):
    def __init__(self, img_size: int, mesh):
        super(MutualProjection, self).__init__()
        self.height = img_size
        self.width = img_size
        self.mutual_trans_mat = MutualTransformation()
        self.ball_render = BallRender(self.width,self.height)

        if type(mesh) == dict:
            radiuses = []
            for bone in mesh['bones']:
                if 'keypoint' in bone:
                    for _, radius in bone['keypoint']:
                        radiuses.append(radius)
            radiuses = torch.tensor(radiuses).type(torch.float32)
        elif type(mesh) == list:
            radiuses = torch.tensor(mesh).type(torch.float32)
        else:
            raise TypeError('mesh can only be list or dict')
        self.num_joints = len(radiuses)
        radiuses = radiuses.view(1, 1, 1, self.num_joints)
        self.register_buffer('radiuses', radiuses)
    
    def forward(self, 
                camera_poses: torch.tensor, 
                inv_camera_poses: torch.tensor, 
                joints: torch.tensor):
        num_batch = joints.shape[0]
        num_views = joints.shape[1]
        num_joints = joints.shape[2]
        joints = joints.view(num_batch, num_views, 1, num_joints, 3, 1)
        joints = joints.repeat(1, 1, num_views, 1, 1, 1)

        mutual_trans_mats = self.mutual_trans_mat(camera_poses, inv_camera_poses)
        mutual_trans_mats = mutual_trans_mats.view(num_batch, num_views, num_views, 1, 4, 4)
        mutual_trans_mats = mutual_trans_mats.repeat(1, 1, 1, num_joints, 1, 1)
        mutual_trans_mats = mutual_trans_mats.detach()

        radiuses = self.radiuses.repeat(num_batch, num_views, num_views, 1)
        projected_points = torch.matmul(mutual_trans_mats[:,:,:,:,0:3,0:3], joints) + mutual_trans_mats[:,:,:,:,0:3,3].unsqueeze(dim=-1)
        squeezed_projected_points = projected_points.view(-1, 3)
        squeezed_radiuses = radiuses.view(-1)
        squeezed_ball_imgs = self.ball_render(squeezed_projected_points, squeezed_radiuses)
        ball_imgs = squeezed_ball_imgs.view(num_batch, num_views, num_views, num_joints, self.height, self.width)
        depth_imgs, _ = ball_imgs.min(dim=3)
        return depth_imgs, projected_points


class MutualProjectionLoss(nn.Module):
    def __init__(self, img_size: int, radiuses: np.ndarray):
        super(MutualProjectionLoss, self).__init__()
        self.mutual_projection = MutualProjection(img_size, radiuses)
        self.data_to_model_criterion = DataToModelLoss(img_size, img_size, radiuses)
        self.model_to_data_criterion = nn.MSELoss()
        self.model_beneth_surface_criterion = nn.MSELoss()
        self.num_joints = len(radiuses)
        self.relu = nn.ReLU()
    
    def forward(self, 
                camera_poses: torch.tensor, 
                inv_camera_poses: torch.tensor, 
                joints: torch.tensor,
                depth_maps: torch.tensor,
                is_mv: bool = True):
        projected_dms, projected_joints = self.mutual_projection(camera_poses, inv_camera_poses, joints)
        num_views = camera_poses.shape[1]
        depth_maps = depth_maps.unsqueeze(dim=1).repeat(1, num_views, 1, 1, 1)

        if is_mv:
            model_to_data_loss = self.model_to_data_criterion(projected_dms, depth_maps) * 9

            depth_maps = depth_maps.view(-1, depth_maps.shape[-2], depth_maps.shape[-1])
            projected_joints = projected_joints.view(-1, self.num_joints, 3)
            data_to_model_loss = self.data_to_model_criterion(depth_maps, projected_joints) * 9

        else:
            model_to_data_loss = 0
            model_to_data_loss += self.model_to_data_criterion(
                projected_dms[:,0,0], depth_maps[:,0,0])
            model_to_data_loss += self.model_to_data_criterion(
                projected_dms[:,1,1], depth_maps[:,1,1])
            model_to_data_loss += self.model_to_data_criterion(
                projected_dms[:,2,2], depth_maps[:,2,2])
            model_to_data_loss *= 3

            data_to_model_loss = 0
            data_to_model_loss += self.data_to_model_criterion(
                depth_maps[:,0,0], projected_joints[:,0,0]
            )
            data_to_model_loss += self.data_to_model_criterion(
                depth_maps[:,1,1], projected_joints[:,1,1]
            )
            data_to_model_loss += self.data_to_model_criterion(
                depth_maps[:,2,2], projected_joints[:,2,2]
            )
            data_to_model_loss *= 3

        loss = model_to_data_loss + data_to_model_loss * 500
        return loss, projected_dms


class MultiviewConsistencyLoss(nn.Module):
    def __init__(self):
        super(MultiviewConsistencyLoss, self).__init__()
        self.loss_func = torch.nn.MSELoss()
    
    def forward(self,
                camera_poses: torch.tensor,
                joints: torch.tensor,
                hm_weight: torch.tensor = None):
        '''camera_poses: B * V * 4 * 4
            joints:      B * V * J * 3
            hm_weight:      B * V * J * 1
        '''
        num_batch = joints.shape[0]
        num_views = joints.shape[1]
        num_joints = joints.shape[2]
        camera_poses = camera_poses.unsqueeze(dim=2)
        camera_poses = camera_poses.repeat(1, 1, num_joints, 1, 1)
        joints = joints.view(num_batch, num_views, num_joints, 3, 1)
        normalized_joints = joints
        canonical_joints = torch.matmul(camera_poses[:,:,:,0:3,0:3], normalized_joints) + camera_poses[:,:,:,0:3,3].unsqueeze(dim=-1)
        loss = 0

        robust_average, indices = torch.median(canonical_joints, dim=1)
        robust_average = robust_average.unsqueeze(dim=1)
        if hm_weight is not None:
            hm_weight = hm_weight.unsqueeze(dim=-1).repeat(1, 1, 1, 3)
            indices = indices.squeeze(dim=-1).unsqueeze(dim=1)
            hm_weight = torch.gather(hm_weight, dim=1, index=indices)
            hm_weight = hm_weight.unsqueeze(dim=-1)
            loss = hm_weight * (robust_average - canonical_joints)**2
            loss = loss.sum()
        else:
            loss = self.loss_func(robust_average, canonical_joints)
        return loss


class WeightedMultiviewConsistencyLoss(nn.Module):
    def __init__(self):
        super(WeightedMultiviewConsistencyLoss, self).__init__()
        self.loss_func = torch.nn.MSELoss()
    
    def forward(self,
                camera_poses: torch.tensor,
                joints: torch.tensor,
                hm_weight: torch.tensor):
        '''camera_poses: B * V * 4 * 4
            joints:      B * V * J * 3
            hm_weight:      B * V * J * 1
        '''
        num_batch = joints.shape[0]
        num_views = joints.shape[1]
        num_joints = joints.shape[2]
        camera_poses = camera_poses.unsqueeze(dim=2)
        camera_poses = camera_poses.repeat(1, 1, num_joints, 1, 1)
        joints = joints.view(num_batch, num_views, num_joints, 3, 1)
        normalized_joints = joints
        canonical_joints = torch.matmul(camera_poses[:,:,:,0:3,0:3], normalized_joints) + camera_poses[:,:,:,0:3,3].unsqueeze(dim=-1)
        canonical_joints = canonical_joints.view(num_batch, num_views, num_joints, 3)

        weights, indices = torch.max(hm_weight, dim=1)
        weights = weights.view(num_batch, 1, num_joints, 1)
        indices = indices.view(num_batch, 1, num_joints, 1).repeat(1, 1, 1, 3)
        average_joints = torch.gather(
            canonical_joints, dim=1, index=indices)
        
        loss = (average_joints - canonical_joints)**2
        loss = loss.sum()
        return loss

class FuseMvPose(nn.Module):
    def __init__(self):
        super(FuseMvPose, self).__init__()
        self.heatmap_variance = HeatmapVariance(16, 16)
    
    def forward(self, joints, camera_poses, inv_camera_poses, uv_hm):
        num_batch = joints.shape[0]
        num_views = joints.shape[1]
        num_joints = joints.shape[2]
        camera_poses = camera_poses.unsqueeze(dim=2)
        camera_poses = camera_poses.repeat(1, 1, num_joints, 1, 1)

        inv_camera_poses = inv_camera_poses.unsqueeze(dim=2)
        inv_camera_poses = inv_camera_poses.repeat(1, 1, num_joints, 1, 1)

        joints = joints.view(num_batch, num_views, num_joints, 3, 1)
        canonical_joints = torch.matmul(camera_poses[:,:,:,0:3,0:3], joints) + camera_poses[:,:,:,0:3,3].unsqueeze(dim=-1)

        heatmap_size = uv_hm.shape[-1]
        uv_hm = uv_hm.reshape(num_batch*num_views, num_joints,    heatmap_size, heatmap_size)
        hm_var = self.heatmap_variance(uv_hm)
        hm_var = hm_var.view(num_batch, num_views, num_joints)
        hm_weight = torch.exp(-10 * hm_var).detach()

        weights, indices = torch.max(hm_weight, dim=1)
        weights = weights.view(num_batch, 1, num_joints, 1)
        indices = indices.view(num_batch, 1, num_joints, 1).repeat(1, 1, 1, 3)
        canonical_joints = canonical_joints.view(num_batch, num_views, num_joints, 3)
        average_joints = torch.gather(
            canonical_joints, dim=1, index=indices)
        average_joints = average_joints.reshape(num_batch, 1, num_joints, 3, 1)

        fused_joints = torch.matmul(inv_camera_poses[:,:,:,0:3,0:3], average_joints) + inv_camera_poses[:,:,:,0:3,3].unsqueeze(dim=-1)
        fused_joints = fused_joints.reshape(num_batch, num_views, num_joints, 3)
        return fused_joints