from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.joint_angle import JointAngleDataset
from mesh.kinematicsTransformation import SkeletonFK
import pickle
from torch.utils.data import DataLoader
import numpy as np


key_points = list(range(11))
input_3d_points = list(range(11,41))
input_2d_points = list(range(11))
input_indices = [i*3 for i in input_3d_points] + [i*3+1 for i in input_3d_points] + [i*3+2 for i in input_3d_points]
input_indices += [i*3 for i in input_2d_points] + [i*3+1 for i in input_2d_points]
output_indices = []
for pt in key_points:
    output_indices += [pt*3, pt*3+1, pt*3+2]

class PoseDenoiser(nn.Module):
    def __init__(self, 
                 input_indices: list = input_indices, 
                 output_indices: list = output_indices, 
                 model_path: str=None):
        super(PoseDenoiser, self).__init__()
        self.input_fea = len(input_indices)
        self.output_fea = len(output_indices)
        self.scale_factor = 0.01

        input_indices = torch.tensor(input_indices).long()
        output_indices = torch.tensor(output_indices).long()

        self.register_buffer('input_indices', input_indices)
        self.register_buffer('output_indices', output_indices)
        self._make_network()
        self.criterion = nn.MSELoss()

        if model_path is not None:
            check_point = torch.load(model_path)
            self.load_state_dict(check_point['network_state_dict'])
            for param in self.parameters():
                param.requires_grad = False

    def _make_network(self):
        self.network = nn.Sequential(
           *[nn.Linear(self.input_fea, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_fea)]
        )
    
    def forward(self, fea):
        is_skel = False
        if fea.ndimension() == 3:
            num_batch = fea.shape[0]
            num_joints = fea.shape[1]
            is_skel = True
            fea = fea.reshape(num_batch, -1)

        input_fea = fea[:, self.input_indices] * self.scale_factor
        if self.training:
            input_fea = input_fea + torch.randn_like(input_fea) * 0.1
        output_fea = self.network(input_fea) / self.scale_factor
        denoised_fea = fea.clone()
        denoised_fea[:, self.output_indices] = output_fea

        if is_skel:
            denoised_fea = denoised_fea.reshape(num_batch, num_joints, 3)
        return denoised_fea

    def loss(self, gt_fea, est_fea):
        num_batch = gt_fea.shape[0]
        gt_fea = gt_fea.view(num_batch, -1)
        est_fea = est_fea.view(num_batch, -1)
        gt_fea = gt_fea[:, self.output_indices]
        est_fea = est_fea[:, self.output_indices]
        return self.criterion(gt_fea, est_fea)


def average_joint_error(gt_joints: torch.tensor, 
                        est_joints: torch.tensor,
                        key_points: list):
    gt_joints = gt_joints.cpu()
    est_joints = est_joints.cpu()

    gt_joints = gt_joints[:, key_points, :]
    est_joints = est_joints[:, key_points, :]

    error = (gt_joints - est_joints).norm(dim=-1)
    return float(error.mean())


    
def train():
    with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    dataset = JointAngleDataset()
    skel_fk = SkeletonFK(mesh)
    pose_denoiser = PoseDenoiser(input_indices, output_indices)
    num_epoch = 5
    batch_size = 128

    import os, random, string
    def rand_model_name():
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
    exp_dir = os.path.join('D:\\exp\\pose_denoiser\\', rand_model_name())
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if torch.cuda.device_count() > 0:
        pose_denoiser = pose_denoiser.cuda()
        skel_fk = skel_fk.cuda()

    optimizer = torch.optim.Adam(pose_denoiser.parameters(), lr=1e-3)
    for epoch in range(num_epoch):
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        total_loss = 0
        total_error = 0
        for num_it, pose_para in enumerate(data_loader):
            if torch.cuda.device_count() > 0:
                pose_para = pose_para.cuda()

            joints = skel_fk(pose_para)[:,:,0:3]
            num_batch = joints.shape[0]
            num_joints=  joints.shape[1]
            joints = joints.reshape(num_batch, -1)
            optimizer.zero_grad()
            denoised_joints = pose_denoiser(joints)
            denoised_joints = denoised_joints.reshape(-1, num_joints, 3)
            joints = joints.reshape(-1, num_joints, 3)
            loss = pose_denoiser.loss(joints, denoised_joints)
            loss.backward()
            total_loss += loss.item()
            error = average_joint_error(joints, denoised_joints, key_points)
            total_error += error
            if num_it %200 == 1:
                print('[{}_{}] curr_loss: {:.5f}, avg_loss: {:.5f}, avg_eror: {:.5f}'.format(epoch, num_it, loss.item(), total_loss / num_it, total_error / num_it))

            optimizer.step()
        
        pth_path = os.path.join(exp_dir, 'model_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'network_state_dict': pose_denoiser.state_dict(),
        }, pth_path)

if __name__ == '__main__':
    train()