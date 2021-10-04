from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from mesh.kinematicsTransformation import HandTransformationMat
from mesh.pointTransformation import RandScale
from mesh.render import DepthRender, Hand3DHeatmapRender


class DepthResample(nn.Module):
    def __init__(self, sample_ratio, kernel_size: int=3):
        super(DepthResample, self).__init__()
        padding = kernel_size // 2
        self.sample_ratio = sample_ratio
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        if kernel_size == 5:
            gaussian_kernel = torch.tensor([
            1, 4, 7, 4, 1,
            4, 16, 26, 16, 4,
            7, 26, 41, 26, 7,
            4, 16, 26, 16, 4,
            1, 4, 7, 4, 1,
            ]).float()
        elif kernel_size == 3:
            gaussian_kernel = torch.tensor([
                1, 2, 1,
                2, 6, 2,
                1, 2, 1
            ]).float()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    
    def forward(self, dm):
        if dm.ndimension() == 3:
            dm = dm.unsqueeze(dim=1)
        samples = torch.rand_like(dm)
        sample_mask = samples > self.sample_ratio
        dm[sample_mask] = 1.0
        dm = self.gaussian_filter(dm)
        return dm


class DepthNoise(nn.Module):
    def __init__(self, width, height):
        super(DepthNoise, self).__init__()
        self.sigma_x = 0.5
        self.sigma_y = 0.5
        self.sigma_z = 0.05
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = torch.from_numpy(u_grid).type(torch.long)
        v_grid = torch.from_numpy(v_grid).type(torch.long)
        u_grid = u_grid.unsqueeze(dim=0)
        v_grid = v_grid.unsqueeze(dim=0)
        self.register_buffer('u_grid', u_grid)
        self.register_buffer('v_grid', v_grid)
    
    def forward(self, dm):
        num_batch = dm.shape[0]
        height = dm.shape[1]
        width = dm.shape[2]
        shift_x = torch.randn_like(dm) * self.sigma_x + 0.5
        shift_x = shift_x.long()
        shift_x = shift_x + self.u_grid.repeat(num_batch, 1, 1)
        shift_x = torch.clamp(shift_x, 0, width-1)

        shift_y = torch.randn_like(dm) * self.sigma_y + 0.5
        shift_y = shift_y.long()
        shift_y = shift_y + self.v_grid.repeat(num_batch, 1, 1)
        shift_y = torch.clamp(shift_y, 0, height-1)

        shift_index = torch.arange(num_batch) * width * height
        if dm.is_cuda:
            shift_index = shift_index.cuda()
        shift_index = shift_index.view(num_batch, 1, 1).repeat(1, height, width)
        shift_index = shift_index + width * shift_y + shift_x

        noise_dm = torch.ones_like(dm)
        noise_dm = dm.view(-1)[shift_index.view(-1)]
        noise_dm = noise_dm.view(num_batch, height, width)
        noise_dm[noise_dm<1.0] += torch.randn_like(noise_dm)[noise_dm<1.0] * self.sigma_z
        return noise_dm

class HandSynthesizer(nn.Module):
    ''' here we assume all the rendered size are square
    '''
    def __init__(self, mesh: dict, image_size: int, heatmap_size: int, uv_hm_scale: float, depth_scale: float, add_noise: bool=True, out_heatmap: bool=True):
        super(HandSynthesizer, self).__init__()
        offset_mats = []
        for bone in mesh['bones']:
            offset_mats.append(bone['offset_matrix'].astype(np.float32))
        self.uv_hm_scale = uv_hm_scale
        self.depth_scale = depth_scale
        self.hand_skeleton_transform = HandTransformationMat(offset_mats)
        self.hm_render = Hand3DHeatmapRender(mesh['bones'], heatmap_size)       
        self.dm_render = DepthRender(mesh, image_size)
        self.rand_scale = RandScale(0.1)
        self.depth_noiser = DepthNoise(image_size, image_size)
        self.add_noise = add_noise
        self.out_heatmap = out_heatmap

    def forward(self, parameters: torch.tensor):
        transform_mats = self.rand_scale(self.hand_skeleton_transform(parameters))
        num_batch = transform_mats.shape[0]
        rand_f_ratio = torch.rand(num_batch)*0.2 + 0.9
        if transform_mats.is_cuda:
            rand_f_ratio = rand_f_ratio.cuda()

        rendered_depth_map = self.dm_render(transform_mats, rand_f_ratio)
        rendered_depth_map = rendered_depth_map * self.depth_scale

        if self.add_noise:
            rendered_depth_map = self.depth_noiser(rendered_depth_map)
        if not self.out_heatmap:
            return rendered_depth_map

        uv_hms, depth_hms, xyz_pts = self.hm_render(transform_mats, rand_f_ratio)
        uv_hms = uv_hms * self.uv_hm_scale
        depth_hms = depth_hms * self.depth_scale
        return rendered_depth_map.detach(), uv_hms.detach(), depth_hms.detach(), xyz_pts.detach()



class SpatialSoftmax(nn.Module):
    def __init__(self, sigma: float = 20.0):
        super(SpatialSoftmax, self).__init__()
        self.sigma = sigma
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, hms):
        assert hms.ndimension() == 4
        num_batch = hms.shape[0]
        num_joint = hms.shape[1]
        height = hms.shape[2]
        width = hms.shape[3]
        hms = hms * self.sigma
        hms = hms.reshape(num_batch*num_joint, height*width)
        softmaxied_hms = self.softmax(hms)
        return softmaxied_hms.reshape(num_batch, num_joint, height, width)


class SpatialNormalization(nn.Module):
    def __init__(self):
        super(SpatialNormalization, self).__init__()
        self.relu = nn.ReLU()
    
    def forward(self, hms):
        assert hms.ndimension() == 4
        # hms = torch.abs(hms)
        hms = self.relu(hms)
        num_batch = hms.shape[0]
        num_joint = hms.shape[1]
        height = hms.shape[2]
        width = hms.shape[3]
        hms = hms.reshape(num_batch, num_joint, height*width)
        sums = hms.sum(dim=-1)
        sums = sums.unsqueeze(dim=-1)
        normalized_hms = hms / (sums + 1e-5)
        return normalized_hms.reshape(num_batch, num_joint, height, width)


class RecoverXYZCoordinateFromHeatmap(nn.Module):
    def __init__(self, width: int, height: int, depth_scale: float):
        super(RecoverXYZCoordinateFromHeatmap, self).__init__()
        self.depth_scale = 1.0 / depth_scale
        self.fx = width / 300.0
        self.fy = height / 300.0
        self.cx = width / 2
        self.cy = height / 2
        self.hm_softmax = SpatialSoftmax(20)
        self.hm_norm = SpatialNormalization()
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = u_grid.reshape((1, 1, height, width))
        v_grid = v_grid.reshape((1, 1, height, width))
        u_grid = torch.from_numpy(u_grid).type(torch.float)
        v_grid = torch.from_numpy(v_grid).type(torch.float)
        self.register_buffer('u_grid', u_grid)
        self.register_buffer('v_grid', v_grid)
    
    def forward(self, uv_hms: torch.tensor, 
                d_hms: torch.tensor, 
                is_shuffing: bool=False):
        normed_uv_hms = self.hm_softmax(uv_hms)
        num_batch = normed_uv_hms.shape[0]
        num_joint = normed_uv_hms.shape[1]
        u_grid = self.u_grid.repeat(num_batch, num_joint, 1, 1)
        v_grid = self.v_grid.repeat(num_batch, num_joint, 1, 1)
        u_hm = normed_uv_hms * u_grid
        v_hm = normed_uv_hms * v_grid
        
        d_hm = d_hms * self.hm_norm(uv_hms)
        u_coors = u_hm.view(num_batch, num_joint, -1).sum(dim=2).unsqueeze(dim=-1)
        v_coors = v_hm.view(num_batch, num_joint, -1).sum(dim=2).unsqueeze(dim=-1)
        d_coors = d_hm.view(num_batch, num_joint, -1).sum(dim=2).unsqueeze(dim=-1)
        x_coors = (u_coors - self.cx) / self.fx
        y_coors = (v_coors - self.cy) / self.fy
        z_coors = d_coors * self.depth_scale
        xyz_coors = torch.cat([x_coors, y_coors, z_coors], dim=-1)
        return xyz_coors

    
class HeatmapVariance(nn.Module):
    def __init__(self, width, height):
        super(HeatmapVariance, self).__init__()
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = (u_grid - width/2) / width
        v_grid = (v_grid - height/2) / height
        u_grid = u_grid.reshape((1, 1, height, width))
        v_grid = v_grid.reshape((1, 1, height, width))
        u_grid = torch.from_numpy(u_grid).type(torch.float)
        v_grid = torch.from_numpy(v_grid).type(torch.float)
        self.register_buffer('u_grid', u_grid)
        self.register_buffer('v_grid', v_grid)
        self.hm_norm = SpatialNormalization()
        self.hm_softmax = SpatialSoftmax(25.0)
    
    def forward(self, hms):
        softmax_hms = self.hm_softmax(hms)
        normed_hms = self.hm_norm(hms)
        num_batch = normed_hms.shape[0]
        num_joint = normed_hms.shape[1]
        # u part
        u_grid = self.u_grid.repeat(num_batch, num_joint, 1, 1)
        u_hm = softmax_hms * u_grid
        u_mean = u_hm.view(num_batch, num_joint, -1).sum(dim=-1)
        u_mean = u_mean.view(num_batch, num_joint, 1, 1)
        u_var = (u_grid - u_mean) ** 2
        u_var = normed_hms * u_var
        u_var = u_var.view(num_batch, num_joint, -1).sum(dim=-1)
        # v part
        v_grid = self.v_grid.repeat(num_batch, num_joint, 1, 1)
        v_hm = softmax_hms * v_grid
        v_mean = v_hm.view(num_batch, num_joint, -1).sum(dim=-1)
        v_mean = v_mean.view(num_batch, num_joint, 1, 1)
        v_var = (v_grid - v_mean) ** 2
        v_var = normed_hms * v_var
        v_var = v_var.view(num_batch, num_joint, -1).sum(dim=-1)
        return u_var + v_var


class PosePriorLoss(nn.Module):
    def __init__(self, pca_mean: np.ndarray, pca_component: np.ndarray):
        super(PosePriorLoss, self).__init__()
        pca_mean = torch.from_numpy(pca_mean).type(torch.float32)
        pca_mean = pca_mean.view(1, pca_mean.shape[0])
        pca_component = torch.from_numpy(pca_component).type(torch.float32)
        pca_space = torch.matmul(pca_component.transpose(0,1), pca_component)
        self.register_buffer('pca_space', pca_space)
        self.register_buffer('pca_mean', pca_mean)
        self.pca_space.requires_grad = False
        self.pca_mean.requires_grad = False
        self.mse_loss = nn.MSELoss()
    
    def forward(self, joints):
        num_batch = joints.shape[0]
        if joints.ndimension() == 4:
            num_view = joints.shape[1]
            num_joints = joints.shape[2]
            skel_root = joints[:,:,0,:].unsqueeze(dim=2)
            joints = joints - skel_root
        else:
            num_view = 1
            num_joints = joints.shape[1]
            skel_root = joints[:,0,:].unsqueeze(dim=1)
            joints = joints - skel_root
        skel_root = joints[:,0,:].unsqueeze(dim=1)
        joints = joints - skel_root
        joints = joints.reshape(num_batch*num_view, num_joints*3)
        joints = joints - self.pca_mean
        recons_joints = torch.matmul(joints, self.pca_space)
        return self.mse_loss(joints, recons_joints)


class PosePriorReconstruction(nn.Module):
    def __init__(self, pca_mean: np.ndarray, pca_component: np.ndarray):
        super(PosePriorReconstruction, self).__init__()
        pca_mean = torch.from_numpy(pca_mean).type(torch.float32)
        pca_mean = pca_mean.view(1, pca_mean.shape[0])
        pca_component = torch.from_numpy(pca_component).type(torch.float32)
        pca_space = torch.matmul(pca_component.transpose(0,1), pca_component)
        self.register_buffer('pca_space', pca_space)
        self.register_buffer('pca_mean', pca_mean)
        self.pca_space.requires_grad = False
        self.pca_mean.requires_grad = False
    
    def forward(self, joints):
        num_batch = joints.shape[0]
        if joints.ndimension() == 4:
            num_view = joints.shape[1]
            num_joints = joints.shape[2]
            skel_root = joints[:,:,0,:].unsqueeze(dim=2)
            joints = joints - skel_root
        else:
            num_view = 1
            num_joints = joints.shape[1]
            skel_root = joints[:,0,:].unsqueeze(dim=1)
            joints = joints - skel_root
        joints = joints.reshape(num_batch*num_view, num_joints*3)
        joints = joints - self.pca_mean
        recons_joints = torch.matmul(joints, self.pca_space)
        recons_joints = recons_joints + self.pca_mean
        recons_joints = recons_joints.reshape(num_batch, num_view, num_joints, 3).squeeze()
        recons_joints = recons_joints + skel_root
        return recons_joints


class DepthSegmentation(nn.Module):
    def __init__(self, width: int, height: int):
        super(DepthSegmentation, self).__init__()
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = u_grid.astype(np.float32)
        v_grid = v_grid.astype(np.float32)
        u_grid = torch.from_numpy(u_grid).float().view(1, 1, height, width)
        v_grid = torch.from_numpy(v_grid).float().view(1, 1, height, width)
        self.register_buffer('u_grid', u_grid)
        self.register_buffer('v_grid', v_grid)
        self.width = width
        self.height = height
        self.fx = self.width / 300
        self.cx = self.width / 2
        self.fy = self.height / 300
        self.cy = self.height / 2
    
    def forward(self, dms: torch.tensor, joints: torch.tensor):
        num_batch = dms.shape[0]
        num_views = dms.shape[1]
        num_joint = joints.shape[2]

        dms = dms.view(-1, self.height, self.width)
        joints = joints.view(-1, num_joint, 3)
        joint_u = joints[:,:,0] * self.fx + self.cx
        joint_v = joints[:,:,1] * self.fy + self.cy
        joint_u = joint_u.view(num_batch*num_views, num_joint, 1, 1)
        joint_v = joint_v.view(num_batch*num_views, num_joint, 1, 1)
        u_grid = self.u_grid.repeat(num_batch*num_views, num_joint, 1, 1)
        v_grid = self.v_grid.repeat(num_batch*num_views, num_joint, 1, 1)

        sq_uv_dist = (joint_u - u_grid)**2 + (joint_v - v_grid)**2
        uv_dist = torch.sqrt(sq_uv_dist)
        min_uv_dist, _ = uv_dist.min(dim=1)
        bg_mask = min_uv_dist > 7
        dms[bg_mask] = 100.0
        dms = dms.view(num_batch, num_views, self.height, self.width)
        return dms.detach()


class ClampedL2lLoss(nn.Module):
    def __init__(self, thresh: float):
        super(ClampedL2lLoss, self).__init__()
        self.thresh = thresh
        self.l2_criterion = nn.MSELoss()

    def forward(self, src: torch.tensor, dst: torch.tensor):
        diff = src - dst
        diff = torch.clamp(diff, min=-self.thresh, max=self.thresh)
        return self.l2_criterion(diff, torch.zeros_like(diff))

class TemporalSmoothnessLoss(nn.Module):
    def __init__(self):
        super(TemporalSmoothnessLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.previous_skel = None
        self.criterion = ClampedL2lLoss(2500)
    
    def forward(self, joints: torch.tensor):
        assert joints.ndimension() == 4
        num_batch = joints.shape[0]

        if self.previous_skel is None:
            prev_joints = joints[:num_batch-1].clone().detach()
            curr_joints = joints[1:]
        else:
            prev_joints = torch.ones_like(joints)
            prev_joints[0] = self.previous_skel.clone().detach()
            prev_joints[1:] = joints[:num_batch-1].clone().detach()
            curr_joints = joints

        self.previous_skel = joints[-1,:].clone().detach()
        return self.criterion(prev_joints, curr_joints)

class ResizeCropImage(nn.Module):
    def __init__(self):
        super(ResizeCropImage, self).__init__()
        self.sigma_z = 0.05
    
    def forward(self, 
                depth_maps: torch.tensor, 
                u_scales: torch.tensor, 
                v_scales: torch.tensor):
        height = depth_maps.shape[-2]
        width = depth_maps.shape[-1]

        cropped_dms = torch.ones_like(depth_maps)
        for idx, (dm, u_scale, v_scale) in enumerate(zip(depth_maps, u_scales, v_scales)):
            dm = dm.view(1, 1, height, width)
            new_size = (int(height*v_scale+0.5), int(width*u_scale+0.5))
            resized_dm = torch.nn.functional.interpolate(dm, new_size)
            if u_scale > 1:
                u_start = 0
                u_end = width
                orig_u_start = (int(width * u_scale + 0.5) - width) // 2
                orig_u_end = orig_u_start + width
            else:
                orig_u_start = 0
                orig_u_end = int(width * u_scale)
                u_start = (width - int(width * u_scale + 0.5)) // 2
                u_end = u_start + orig_u_end
            
            if v_scale > 1:
                v_start = 0
                v_end = height
                orig_v_start = (int(height * v_scale + 0.5) - height) // 2
                orig_v_end = orig_v_start + height
            else:
                orig_v_start = 0
                orig_v_end = int(height * v_scale)
                v_start = (height - int(height * v_scale + 0.5)) // 2 
                v_end = v_start + orig_v_end

                cropped_dms[idx, v_start:v_end, u_start:u_end] =\
                resized_dm[0,0,orig_v_start:orig_v_end, orig_u_start:orig_u_end]
        return cropped_dms.squeeze()
