from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
from mesh.cuda_kernel import depth_rasterization
from mesh.pointTransformation import LinearBlendSkinning, OthographicalProjection, RandOthographicalProjection, InverseOthographicalProjection


class BallRender(nn.Module):
    def __init__(self, width: int, height: int) -> None:
        super(BallRender, self).__init__()
        self.width = width
        self.height = height
        x_grid, y_grid = np.meshgrid(
            np.arange(self.width), np.arange(self.height))
        self.register_buffer(
            'x_grid', torch.from_numpy(x_grid).type(torch.float))
        self.register_buffer(
            'y_grid', torch.from_numpy(y_grid).type(torch.float))
        self.dist_weight = torch.nn.Parameter(torch.Tensor(1))
    
    def reset_parameters(self):
        self.dist_weight = 0

    def forward(self, xyz_centers, radiuses):
        num_batch = xyz_centers.shape[0]
        x_grid = self.x_grid.unsqueeze(0).repeat(num_batch, 1, 1)
        y_grid = self.y_grid.unsqueeze(0).repeat(num_batch, 1, 1)

        x_grid = (x_grid - self.width/2) * 300.0 / self.width
        y_grid = (y_grid - self.height/2) * 300.0 / self.height

        x = xyz_centers[:,0].view(num_batch, 1, 1)
        y = xyz_centers[:,1].view(num_batch, 1, 1)
        z = xyz_centers[:,2].view(num_batch, 1, 1)
        x_squared_dist = (x_grid - x)**2
        y_squared_dist = (y_grid - y)**2
        z_dist = z.repeat(1, self.height, self.width)
        radiuses = radiuses.view(num_batch, 1, 1)
        squared_dist_map = torch.clamp(radiuses**2 - x_squared_dist - y_squared_dist, min=1e-2)
        surface_mask = squared_dist_map != 1e-2
        backgroud_mask = squared_dist_map == 1e-2
        dist_map = torch.sqrt(squared_dist_map)
        
        depth_map = torch.zeros_like(dist_map)
        depth_map[surface_mask] = z_dist[surface_mask] - dist_map[surface_mask]

        # uv_squared_dist_map = x_squared_dist + y_squared_dist
        # exp_dist_map = torch.exp(-0.01*uv_squared_dist_map)
        # depth_map[backgroud_mask] = 100.0 - (100 - z_dist[backgroud_mask]) *exp_dist_map[backgroud_mask]
        depth_map[backgroud_mask] = 100.0
        return depth_map


class HandBallPrimitiveRender(nn.Module):
    def __init__(self, bones: list, width: int, height: int) -> None:
        super(HandBallPrimitiveRender, self).__init__()
        self.width = width
        self.height = height
        self.ball_renderer = BallRender(self.width, self.height)
        vertices, radiuses = [], []
        skinning_weights, skinning_vertex_indices = [], []

        for bone in bones:
            skinning_weights.append([])
            skinning_vertex_indices.append([])
            if 'keypoint' in bone:
                for pt, radius in bone['keypoint']:
                    pt = np.asarray([pt[0], pt[1], pt[2], 1.0], np.float32)
                    vertices.append(pt)
                    radiuses.append(radius)
                    skinning_weights[-1].append(1.0)
                    skinning_vertex_indices[-1].append(len(vertices)-1)
        vertices = np.asarray(vertices).astype(np.float32)
        self.num_vertices = len(vertices)
        self.lbs = LinearBlendSkinning(vertices, skinning_weights, skinning_vertex_indices)
        radiuses = torch.tensor(radiuses).type(torch.float).unsqueeze(0)
        self.register_buffer('radiuses', radiuses)
    
    def forward(self, transformation_mats: torch.tensor):
        skinned_points = self.lbs(transformation_mats)
        num_batch = skinned_points.shape[0]
        radiuses = self.radiuses.repeat(num_batch, 1)
        skinned_points = skinned_points.view(-1, 4)
        radiuses = radiuses.view(-1)
        balls = self.ball_renderer(skinned_points, radiuses)
        part_maps = balls.view(num_batch, self.num_vertices, self.height, self.width)
        depth_maps, _ = torch.min(part_maps, dim=1)
        return part_maps, depth_maps


class DataToModelLoss(nn.Module):
    def __init__(self, width, height, mesh):
        super(DataToModelLoss, self).__init__()
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = u_grid.astype(np.float32)
        v_grid = v_grid.astype(np.float32)
        x_grid = (u_grid - width / 2) * 300.0 / width
        y_grid = (v_grid - height / 2) * 300.0 / height
        x_grid = torch.from_numpy(x_grid).float().unsqueeze(dim=0)
        y_grid = torch.from_numpy(y_grid).float().unsqueeze(dim=0)
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)
        self.width = width
        self.height = height
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
        self.relu = nn.ReLU()
    
    def forward(self, dms: torch.tensor, joints: torch.tensor):
        num_batch = dms.shape[0]

        z = dms.view(num_batch, 1, self.height, self.width, 1)
        x = self.x_grid.repeat(num_batch, 1, 1).view(num_batch, 1, self.height, self.width, 1)
        y = self.y_grid.repeat(num_batch, 1, 1).view(num_batch, 1, self.height, self.width, 1)
        xyz = torch.cat([x,y,z], dim=-1)
        joints = joints.view(num_batch, self.num_joints, 1, 1, 3)

        radiuses = self.radiuses.view(1, self.num_joints, 1, 1)
        distance_to_center = (xyz - joints).norm(dim=-1)
        # distance_to_surface = self.relu(distance_to_center - radiuses)
        distance_to_surface = torch.abs(distance_to_center - radiuses)

        d = dms.view(num_batch, 1, self.height, self.width).repeat(1, self.num_joints, 1, 1)
        background = d>99
        distance_to_surface[background] = 0
        distance_to_surface, _ = distance_to_surface.min(dim=1)
        distance_to_surface = torch.clamp(distance_to_surface, min=0, max=50)
        return distance_to_surface.mean()


class CollisionLoss(nn.Module):
    def __init__(self, min_dist: float=6):
        super(CollisionLoss, self).__init__()
        self.relu = nn.ReLU()
        self.min_sq_dist = min_dist**2
        joint_1 = []
        joint_2 = []
        # every finger to the palm
        for j_1 in range(11):
            for j_2 in range(11, 41):
                joint_1.append(j_1)
                joint_2.append(j_2)
        # every finger to the rest
        for j_1 in range(11, 41):
            for j_2 in range(j_1+1, 41):
                if (j_1 - 11) // 6 != (j_2 - 11) // 6:
                    joint_1.append(j_1)
                    joint_2.append(j_2)
        joint_1 = torch.tensor(joint_1).long()
        joint_2 = torch.tensor(joint_2).long()
        self.register_buffer('joint_1', joint_1)
        self.register_buffer('joint_2', joint_2)
    
    def forward(self, joints):
        num_batch = joints.shape[0]
        joints = joints.view(num_batch, -1, 3)
        joints_1 = joints[:, self.joint_1]
        joints_2 = joints[:, self.joint_2]
        sq_dist = (joints_1 - joints_2)**2
        sq_dist = sq_dist.sum(dim=-1)
        collision_loss = self.relu(self.min_sq_dist - sq_dist)
        return collision_loss.sum()

        
class BoneLengthLoss(nn.Module):
    def __init__(self):
        super(BoneLengthLoss, self).__init__()
        import mesh.bone_length
        joint_1 = torch.tensor(mesh.bone_length.joint_1).long()
        joint_2 = torch.tensor(mesh.bone_length.joint_2).long()
        median_length = torch.tensor(mesh.bone_length.uniform_length).float()
        min_sq_length = (median_length * 0.80)**2
        min_sq_length = min_sq_length.unsqueeze(dim=0)
        max_sq_length = (median_length * 1.05)**2
        max_sq_length = max_sq_length.unsqueeze(dim=0)
        self.register_buffer('joint_1', joint_1)
        self.register_buffer('joint_2', joint_2)
        self.register_buffer('max_length', max_sq_length)
        self.register_buffer('min_length', min_sq_length)
        self.relu = nn.ReLU()
    
    def forward(self, joints):
        num_batch = joints.shape[0]
        joints = joints.view(num_batch, -1, 3)
        joints_1 = joints[:, self.joint_1]
        joints_2 = joints[:, self.joint_2]
        sq_dist = (joints_1 - joints_2)**2
        sq_dist = sq_dist.sum(dim=-1)
        lower_loss = self.relu(self.min_length - sq_dist).mean()
        uppper_loss = self.relu(sq_dist - self.max_length).mean()
        loss = lower_loss + uppper_loss
        return loss



class HeatmapRender(nn.Module):
    def __init__(self, 
                 hm_size: int,
                 sigma: float = 1.0):
        super(HeatmapRender, self).__init__()
        self.sigma = sigma
        self.height = hm_size
        self.width = hm_size
        u_grid, v_grid = np.meshgrid(np.arange(self.width), np.arange(self.height))
        u_grid = np.expand_dims(u_grid, axis=0)
        v_grid = np.expand_dims(v_grid, axis=0)
        u_grid = torch.from_numpy(u_grid).type(torch.float32)
        v_grid = torch.from_numpy(v_grid).type(torch.float32)
        self.register_buffer('u_grid', u_grid)
        self.register_buffer('v_grid', v_grid)

    def forward(self, uvd_points: torch.tensor):
        assert uvd_points.ndimension() == 3
        num_batch = uvd_points.shape[0]
        num_joint = uvd_points.shape[1]
        num_point = num_batch * num_joint
        uvd_points = uvd_points.view(num_point, -1) # 3d or homographic coordinate
        u_grid = self.u_grid.repeat(num_point, 1, 1)
        v_grid = self.v_grid.repeat(num_point, 1, 1)
        u_pt = uvd_points[:,0].view(num_point, 1, 1)
        v_pt = uvd_points[:,1].view(num_point, 1, 1)
        u_squared_dist = (u_grid - u_pt)**2
        v_squared_dist = (v_grid - v_pt)**2
        uv_hm = torch.exp(-0.5*self.sigma*(u_squared_dist + v_squared_dist))
        uv_mask = uv_hm > 0.05

        d_hm = torch.ones_like(u_grid)
        d_pt = uvd_points[:,2].view(num_point, 1, 1)
        scaled_d_hm = d_hm * d_pt
        scaled_d_hm[~uv_mask] = 0.0

        uv_hm = uv_hm.view(num_batch, num_joint, self.height, self.width)
        scaled_d_hm = scaled_d_hm.view(num_batch, num_joint, self.height, self.width)
        return uv_hm, scaled_d_hm


class Hand3DHeatmapRender(nn.Module):
    def __init__(self, bones: list, heatmap_size: int) -> None:
        super(Hand3DHeatmapRender, self).__init__()
        self.width = heatmap_size
        self.height = heatmap_size
        self.hm_renderer = HeatmapRender(heatmap_size)
        self.camera = OthographicalProjection(self.width/2, self.height/2, self.width/300, self.height/300)
        self.inv_camera = InverseOthographicalProjection(self.width/2, self.height/2, self.width/300, self.height/300)
        vertices = []
        skinning_weights, skinning_vertex_indices = [], []
        for bone in bones:
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
    
    def forward(self, transformation_mats: torch.tensor, rand_f: torch.tensor = None):
        xyz_points = self.lbs(transformation_mats)
        uvd_points = self.camera(xyz_points, rand_f)
        hms, dms = self.hm_renderer(uvd_points)
        xyz_points = self.inv_camera(uvd_points)
        return hms, dms, xyz_points


class DepthRasterizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, width: int, height: int, face_vertices: torch.Tensor):
        depth_maps = depth_rasterization.forward(width, height, face_vertices)
        depth_maps = torch.clamp(depth_maps, max=100.0)
        return depth_maps

class DepthRasterization(torch.nn.Module):
    def __init__(self, 
                 width: int,
                height: int, 
                np_faces: np.ndarray, 
                right_hand:bool = True):
        super(DepthRasterization, self).__init__()
        self.width = width
        self.height = height
        if right_hand:
            orig_np_faces = np_faces.copy()
            np_faces[:,0], np_faces[:,1] = orig_np_faces[:,1], orig_np_faces[:,0]
        faces = torch.from_numpy(np_faces).type(torch.int64).view(-1)

        self.register_buffer('faces', faces)
        self.num_faces = len(np_faces)

    def forward(self, vertices: torch.Tensor):
        num_batch = vertices.shape[0]
        face_vertices = vertices[:, self.faces, 0:3]
        face_vertices = face_vertices.view(num_batch, self.num_faces, 3, 3)
        rendered_dm = DepthRasterizationFunction.apply(640, 640, face_vertices).unsqueeze(dim=1)
        resized_dm = torch.nn.functional.interpolate(rendered_dm, size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(dim=1)
        return resized_dm


class DepthRender(torch.nn.Module):
    def __init__(self, mesh: dict, image_size: int):
        super(DepthRender, self).__init__()
        skinning_weights = []
        skinning_vertex_indices = []
        for bone in mesh['bones']:
            skinning_vertex_indices.append(bone['weight_vertexid'])
            skinning_weights.append(bone['weight_coeff'])
        self.lbs = LinearBlendSkinning(
            mesh['vertices'], skinning_weights, skinning_vertex_indices)
        self.camera = OthographicalProjection(320, 320, 640/300, 640/300)
        self.rasterizer = DepthRasterization(image_size, image_size, mesh['faces'])
    
    def forward(self, transformation_mats: torch.tensor, rand_fx: torch.tensor=None):
        skinned_points = self.camera(self.lbs(transformation_mats), rand_fx)
        depth_maps = self.rasterizer(skinned_points)
        return depth_maps


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    import time
    with open('data/preprocessed_right_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    
    '''
    scale_ratio = 640.0 / 300.0
    faces = mesh['faces']
    vertices = mesh['vertices']
    bones = mesh['bones']

    depth_rasterizer = DepthRasterization(640, 640, faces)
    vertices = np.expand_dims(vertices, axis=0)
    vertices = torch.from_numpy(vertices).type(torch.float32).repeat(1, 1, 1)
    vertices = vertices * 640 / 300 + 320

    transform = torch.eye(4).view(1, 1, 4, 4).repeat(1, 10144, 1 ,1)
    vertices = torch.matmul(transform, vertices)

    depth_rasterizer = depth_rasterizer.cuda()
    vertices = vertices.cuda()
    print(vertices.shape)

    t0 = time.time()
    for _ in range(1):
        depth_maps = depth_rasterizer(vertices)
        print('rendered with %f ms'%(time.time() - t0))
        t0 = time.time()

    depth_maps = depth_rasterizer(vertices)
    depth_maps = depth_maps.cpu().numpy()

    for dm in depth_maps:
        dm[dm>500] = 0.0
        plt.imshow(dm)
        plt.show()
    '''

    ball_render = BallRender(64, 64)
    centers = torch.randn(100, 3) + 32
    radiuses = torch.ones(100) * 10

    depth_map = ball_render(centers, radiuses)
    dm = depth_map[0].numpy()
    import matplotlib.pyplot as plt
    plt.imshow(dm)
    plt.show()        