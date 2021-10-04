from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from math import pi


class LinearBlendSkinning(nn.Module):
    def __init__(
            self,
            vertices: np.ndarray,
            skinning_weights: list,
            skinning_vertex_indices: list,
            right_hand: bool = True):
        super(LinearBlendSkinning, self).__init__()
        assert len(skinning_vertex_indices) == len(
            skinning_weights), 'vertex index and weight should be with the same size'
        for vert_idx, vert_weight in zip(skinning_vertex_indices, skinning_weights):
            assert len(vert_idx) == len(
                vert_weight), 'vertex index and weight should be the same size'

        num_vertices = vertices.shape[0]
        skin_vertices = []
        for weights, indices in zip(skinning_weights, skinning_vertex_indices):
            cur_skin_vert = np.zeros((num_vertices, 4), dtype=np.float32)
            for w, i in zip(weights, indices):
                cur_skin_vert[i] = w * vertices[i]
            cur_skin_vert = torch.from_numpy(cur_skin_vert).type(torch.float)
            skin_vertices.append(cur_skin_vert)
        skin_vertices = [v.unsqueeze(dim=0) for v in skin_vertices]
        skin_vertices = torch.cat(skin_vertices, dim=0)
        skin_vertices = skin_vertices.unsqueeze(dim=0).unsqueeze(dim=-1)
        self.register_buffer('skin_vertices', skin_vertices)
        self.right_hand = right_hand

    def forward(self, bone_transformations):
        if bone_transformations.ndimension() == 4:
            bone_transformations = bone_transformations.unsqueeze(dim=2)
        skinned_vertices = torch.matmul(
            bone_transformations, self.skin_vertices).sum(dim=1).squeeze(dim=-1)
        if self.right_hand:
            skinned_vertices[:,:,0] *= -1
        return skinned_vertices


class RandOthographicalProjection(nn.Module):
    def __init__(self, cx, cy, f_min, f_max):
        assert f_min < f_max, 'f_min should less than f_max'
        super(RandOthographicalProjection, self).__init__()
        self.cx = cx
        self.cy = cy
        self.f_center = f_min + (f_max - f_min) / 2
        self.f_range = f_max - f_min

    def forward(self, xyz_points):
        batch_size = xyz_points.shape[0]
        num_vertices = xyz_points.shape[1]
        xyz_points = xyz_points.view(-1, 4, 1)
        rand_f = (torch.rand_like(xyz_points[:,0,:]) * 2 - 1) * self.f_range + self.x_center
        xyz_points[:,0,:] = xyz_points[:,0,:] * rand_f + self.cx
        xyz_points[:,1,:] = xyz_points[:,1,:] * rand_f + self.cy
        xyz_points = xyz_points.view(batch_size, num_vertices, 4)
        return xyz_points


class OthographicalProjection(nn.Module):
    def __init__(self, cx, cy, fx, fy):
        super(OthographicalProjection, self).__init__()
        k_mat = torch.eye(4)
        k_mat[0, 0] = fx
        k_mat[1, 1] = fy
        k_mat[0, 3] = cx
        k_mat[1, 3] = cy
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        k_mat = k_mat.unsqueeze(0).type(torch.float32)
        self.register_buffer('k_mat', k_mat)

    def forward(self, xyz_points, rand_f = None):
        batch_size = xyz_points.shape[0]
        num_vertices = xyz_points.shape[1]
        xyz_points = xyz_points.view(-1, 4, 1)
        if rand_f is None:
            projected_points = torch.matmul(self.k_mat, xyz_points)
        else:
            rand_f = rand_f.view(-1, 1)
            projected_points = torch.ones_like(xyz_points)
            projected_points = projected_points.view(batch_size, -1, 4)
            xyz_points = xyz_points.view(batch_size, -1, 4)
            projected_points[:,:,0] = xyz_points[:,:,0]*rand_f*self.fx + self.cx
            projected_points[:,:,1] = xyz_points[:,:,1]*rand_f*self.fy + self.cy
            projected_points[:,:,2] = xyz_points[:,:,2]
        projected_points = projected_points.view(batch_size, num_vertices, 4)
        return projected_points


class InverseOthographicalProjection(nn.Module):
    def __init__(self, cx, cy, fx, fy):
        super(InverseOthographicalProjection, self).__init__()
        k_mat = torch.eye(4)
        k_mat[0, 0] = fx
        k_mat[1, 1] = fy
        k_mat[0, 3] = cx
        k_mat[1, 3] = cy
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        inv_k_mat = torch.inverse(k_mat)
        inv_k_mat = inv_k_mat.unsqueeze(0).type(torch.float32)
        self.register_buffer('inv_k_mat', inv_k_mat)

    def forward(self, uvd_points):
        batch_size = uvd_points.shape[0]
        num_vertices = uvd_points.shape[1]
        uvd_points = uvd_points.view(-1, 4, 1)
        xyz_points = torch.matmul(self.inv_k_mat, uvd_points)
        xyz_points = xyz_points.view(batch_size, num_vertices, 4)
        return xyz_points



class RandScale(nn.Module):
    def __init__(self, rand_scale):
        super(RandScale, self).__init__()
        self.rand_scale = rand_scale
        self.register_buffer('identity_mat', 
                              torch.eye(4).unsqueeze(dim=0).type(torch.float))
    
    def forward(self, transform_mats):
        batch_size = transform_mats.shape[0]
        num_transform = transform_mats.shape[1]
        scale_mats = self.identity_mat.repeat(batch_size, 1, 1)

        rand_x_scale = torch.rand(batch_size) * self.rand_scale + 0.90 - self.rand_scale/2
        rand_y_scale = torch.rand(batch_size) * self.rand_scale + 0.90 - self.rand_scale/2
        rand_z_scale = torch.rand(batch_size) * self.rand_scale + 0.90 - self.rand_scale/2

        scale_mats[:, 0, 0] = rand_x_scale
        scale_mats[:, 1, 1] = rand_y_scale
        scale_mats[:, 2, 2] = rand_z_scale
        scale_mats = scale_mats.unsqueeze(dim=1).repeat(1, num_transform, 1, 1)
        return torch.matmul(scale_mats, transform_mats)

