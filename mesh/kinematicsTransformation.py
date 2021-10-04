from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from math import pi
from mesh.pointTransformation import LinearBlendSkinning, RandScale

class AxisRotationMatrix(nn.Module):
    def __init__(self, rotation_axis: np.ndarray):
        super(AxisRotationMatrix, self).__init__()
        x = torch.tensor(rotation_axis[0]).type(torch.float32)
        y = torch.tensor(rotation_axis[1]).type(torch.float32)
        z = torch.tensor(rotation_axis[2]).type(torch.float32)
        self.register_buffer('identity_mat', torch.eye(4).unsqueeze(0))
        self.register_buffer('x', x)
        self.register_buffer('y', y)
        self.register_buffer('z', z)
        self.register_buffer('xx', x*x)
        self.register_buffer('yy', y*y)
        self.register_buffer('zz', z*z)
        self.register_buffer('xy', x*y)
        self.register_buffer('xz', x*z)
        self.register_buffer('yz', y*z)


    def forward(self, angles):
        num_batch = len(angles)
        rot_mats = self.identity_mat.repeat(num_batch, 1, 1)
        ''' rotation matrix from rotation around axis
            see https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
             [[cos+self.xx*(1-cos), self.xy*(1-cos)-self.z*sin, self.xz*(1-cos)+self.y*sin, 0.0],
              [self.xy*(1-cos)+self.z*sin, cos+self.yy*(1-cos), self.yz*(1-cos)-self.x*sin, 0.0],
              [self.xz*(1-cos)-self.y*sin, self.yz*(1-cos)+self.x*sin, cos+self.zz*(1-cos), 0.0],
              [0.0, 0.0, 0.0, 1.0]]
        '''
        c = torch.cos(angles)
        s = torch.sin(angles)
        i = 1 - c

        rot_mats[:,0,0] = self.xx * i + c
        rot_mats[:,0,1] = self.xy * i - self.z * s
        rot_mats[:,0,2] = self.xz * i + self.y * s

        rot_mats[:,1,0] = self.xy * i + self.z * s
        rot_mats[:,1,1] = self.yy * i + c
        rot_mats[:,1,2] = self.yz * i - self.x * s

        rot_mats[:,2,0] = self.xz * i - self.y * s
        rot_mats[:,2,1] = self.yz * i + self.x * s
        rot_mats[:,2,2] = self.zz * i + c
        return rot_mats
    
class  TranslationMatrix(nn.Module):
    def __init__(self):
        super(TranslationMatrix, self).__init__()
        self.register_buffer('identity_mat', torch.eye(4).unsqueeze(0))
    
    def forward(self, translation_params):
        batch_num = translation_params.shape[0]
        assert translation_params.shape[1] == 3, 'number of translation parameters should be 3'
        translation_mat = self.identity_mat.repeat(batch_num, 1, 1)
        translation_mat[:,0,3] = translation_params[:,0]
        translation_mat[:,1,3] = translation_params[:,1]
        translation_mat[:,2,3] = translation_params[:,2]
        return translation_mat

class ScaleMatrix(nn.Module):
    def __init__(self):
        super(ScaleMatrix, self).__init__()
        self.register_buffer('identity_mat', torch.eye(4).unsqueeze(0))
    
    def forward(self, scale):
        num_batch = len(scale)
        scale_mat = self.identity_mat.repeat(num_batch, 1, 1)
        scale_mat[:, 0, 0] = scale
        scale_mat[:, 1, 1] = scale
        scale_mat[:, 2, 2] = scale
        return scale_mat
        
class FingerJoint(nn.Module):
    def __init__(self, np_offset_mat:np.ndarray, rot_axises:list):
        super(FingerJoint, self).__init__()
        offset_mat = torch.from_numpy(np_offset_mat)
        rest_transformation = torch.inverse(offset_mat)
        self.register_buffer('offset_mat', offset_mat)
        self.register_buffer('transform_mat', rest_transformation)
        self.axis_rotation = nn.ModuleList([AxisRotationMatrix(rot_axis) for rot_axis in rot_axises])
    
    def forward(self, joint_angles, parent_transform):
        if joint_angles.ndimension() == 1:
            joint_angles = joint_angles.unsqueeze(1)
        local_transform_mat = None
        for idx, axis_rotation in enumerate(self.axis_rotation):
            angle = joint_angles[:, idx]
            if angle.ndimension() > 2:
                angle = angle.squeeze()
            if local_transform_mat is None:
                local_transform_mat = axis_rotation(angle)
            else:
                local_transform_mat = torch.matmul(local_transform_mat, axis_rotation(angle))
        
        num_batch = joint_angles.shape[0]
        offset_mat = self.offset_mat.unsqueeze(0).repeat(num_batch, 1, 1).detach()
        transform_mat = self.transform_mat.unsqueeze(0).repeat(num_batch, 1, 1).detach()
        global_transform = torch.matmul(transform_mat, local_transform_mat)
        global_transform = torch.matmul(global_transform, offset_mat)
        if parent_transform is not None:
            global_transform = torch.matmul(parent_transform, global_transform)
        return global_transform

class Finger(nn.Module):
    def __init__(self, offset_mats:list, abduct_axis: np.ndarray):
        super(Finger, self).__init__()
        assert len(offset_mats) == 3, 'every finger should contain 3 joints'
        x_axis = np.asarray([1, 0, 0], np.float)
        self.j1 = FingerJoint(offset_mats[0], [abduct_axis, x_axis])
        self.j2 = FingerJoint(offset_mats[1], [x_axis])
        self.j3 = FingerJoint(offset_mats[2], [x_axis])
    
    def forward(self, angles, parent_transform):
        j1_transform = self.j1(angles[:,0:2], parent_transform) 
        j2_transform = self.j2(angles[:,2], j1_transform)
        j3_transform = self.j3(angles[:,3], j2_transform)
        return [j1_transform, j2_transform, j3_transform]

class Palm(nn.Module):
    def __init__(self, np_offset_mat: np.ndarray):
        super(Palm, self).__init__()
        x_axis = np.asarray([1, 0, 0], np.float)
        y_axis = np.asarray([0, 1, 0], np.float)
        z_axis = np.asarray([0, 0, 1], np.float)
        self.x_rotation = AxisRotationMatrix(x_axis)
        self.y_rotation = AxisRotationMatrix(y_axis)
        self.z_rotation = AxisRotationMatrix(z_axis)
        self.translation = TranslationMatrix()
        offset_mat = torch.from_numpy(np_offset_mat)
        rest_transformation = torch.inverse(offset_mat)
        self.scale_mat = ScaleMatrix()
        self.register_buffer('offset_mat', offset_mat)
        self.register_buffer('transform_mat', rest_transformation)
    
    def forward(self, parameters):
        assert parameters.shape[1] == 6, 'palm is of 6 dof'
        num_batch = parameters.shape[0]
        rotation_mat = self.x_rotation(parameters[:,0])
        rotation_mat = torch.matmul(self.y_rotation(parameters[:,1]), rotation_mat)
        rotation_mat = torch.matmul(self.z_rotation(parameters[:,2]), rotation_mat)
        translation_mat = self.translation(parameters[:,3:])
        local_transform = torch.matmul(translation_mat, rotation_mat)
        global_transform = local_transform
        carpals_transform_mat = global_transform
        return [global_transform, carpals_transform_mat]

class HandTransformationMat(nn.Module):
    def __init__(self, offset_mats):
        super(HandTransformationMat, self).__init__()
        self.palm = Palm(offset_mats[0])
        self.fingers = nn.ModuleList()
        y_axis = np.asarray([0, -1, 0], np.float)
        z_axis = np.asarray([0, 0, 1], np.float)
        abduct_axises = [z_axis, z_axis, y_axis, y_axis, z_axis]
        for finger_idx in range(5):
            self.fingers.append(Finger(offset_mats[2+finger_idx*3:2+finger_idx*3+3], abduct_axises[finger_idx]))
        self.fingers = self.fingers

    def forward(self, parameters):
        transform_mats = []
        transform_mats += self.palm(parameters[:,0:6])
        palm_transform_mat = transform_mats[0]
        for finger_idx in range(5):
            transform_mats += self.fingers[finger_idx](parameters[:, 6+4*finger_idx:6+4*finger_idx+4], palm_transform_mat)
        
        transform_mats = torch.cat([m.unsqueeze(1) for m in transform_mats], dim=1)
        return transform_mats


class SkeletonFK(nn.Module):
    def __init__(self, mesh: dict):
        super(SkeletonFK, self).__init__()
        offset_mats = []
        for bone in mesh['bones']:
            offset_mats.append(bone['offset_matrix'].astype(np.float32))
        self.hand_skeleton_transform = HandTransformationMat(offset_mats)
        self.rand_scale = RandScale(0.2)

        vertices = []
        skinning_weights, skinning_vertex_indices = [], []
        for bone in mesh['bones']:
            skinning_weights.append([])
            skinning_vertex_indices.append([])
            if 'keypoint' in bone:
                for pt, _ in bone['keypoint']:
                    pt = np.asarray([pt[0], pt[1], pt[2], 1.0], np.float32)
                    vertices.append(pt)
                    skinning_weights[-1].append(1.0)
                    skinning_vertex_indices[-1].append(len(vertices)-1)
        vertices = np.asarray(vertices).astype(np.float32)
        self.num_vertices = len(vertices)
        self.lbs = LinearBlendSkinning(vertices, skinning_weights, skinning_vertex_indices)
    
    def forward(self, para: torch.tensor):
        transform_mats = self.rand_scale(self.hand_skeleton_transform(para))
        key_points = self.lbs(transform_mats)
        return key_points
