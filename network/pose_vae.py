from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.joint_angle import JointAngleDataset
from mesh.kinematicsTransformation import SkeletonFK
import pickle
from torch.utils.data import DataLoader
import numpy as np

class PoseVae(nn.Module):
    def __init__(self, pose_fea, latent_fea, model_path = None):
        super(PoseVae, self).__init__()
        self.pose_fea = pose_fea
        self.latent_fea = latent_fea
        self._make_encoder()
        self._make_decoder()

        if model_path is not None:
            check_point = torch.load(model_path)
            self.load_state_dict(check_point['network_state_dict'])
            for param in self.parameters():
                param.requires_grad = False

    def _make_encoder(self):
        self.base = nn.Sequential(
           *[nn.Linear(self.pose_fea, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU()]
        )
        self.mu = nn.Linear(256, self.latent_fea)
        self.logvar = nn.Linear(256, self.latent_fea)

    
    def _make_decoder(self):
        self.decoder = nn.Sequential(
            *[nn.Linear(self.latent_fea, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, self.pose_fea)]
        )

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) * 0.1
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    

    def _likelihood(self, x, recon_x, mu, logvar):
        recon = F.mse_loss(x, recon_x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + KLD

    def forward(self, x, do_reparameterize: bool = False):
        base = self.base(x)
        mu = self.mu(base)
        logvar = self.logvar(base)
        if do_reparameterize:
            z = self._reparameterize(mu, logvar)
        else:
            z = mu
        recon_x = self.decoder(z)
        likelihood = self._likelihood(x, recon_x, mu, logvar)
        return recon_x, mu, logvar, likelihood

    def random_sample(self, n: int):
        z = torch.randn(n, self.latent_fea).cuda()
        recon_x = self.decoder(z)
        return recon_x
    
    def prior_loss(self, x):
        x = x.view(-1, self.pose_fea)
        base = self.base(x)
        mu = self.mu(base)
        logvar = self.logvar(base)
        z = self._reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        likelihood = self._likelihood(x, recon_x, mu, logvar)
        return likelihood
    
    def recons(self, x):
        num_batch = x.shape[0]
        num_view = x.shape[1]
        x = x.view(-1, self.pose_fea)
        base = self.base(x)
        mu = self.mu(base)
        recon_x = self.decoder(mu)
        recon_x = recon_x.view(num_batch, num_view, -1, 3)
        return recon_x

import cv2
import numpy as np
joint_color = [(255, 0, 0)] * 11 +\
                  [(25, 255, 25)]*6 +\
                  [(212, 0, 255)]*6 +\
                  [(0, 230, 230)]*6 +\
                  [(179, 179, 0)]*6 +\
                  [(255, 153, 153)]*6

def vis_joint(joint):
    img = np.ones((64,64,3))*255
    img = img.astype(np.uint8)
    joint = joint * 64.0 / 300.0 + 32.0
    for idx, j in enumerate(joint):
        cv2.circle(img, (j[0], j[1]), 2, joint_color[idx], thickness=-1)
    return img

def vis_vae_recons(orig_joints, recon_joints):
    orig_joints = orig_joints * 100.0
    recon_joints = recon_joints * 100.0

    orig_joints = orig_joints.view(-1, 41, 3).detach().cpu().numpy()
    recon_joints = recon_joints.view(-1, 41, 3).detach().cpu().numpy()

    images = []
    for oj, rj in zip(orig_joints, recon_joints):
        images.append(np.hstack([vis_joint(oj), vis_joint(rj)]))
    return np.vstack(images)

def vis_vae_sample(joints):
    joints = joints * 100.0

    joints = joints.view(-1, 41, 3).detach().cpu().numpy()

    images = []
    for j in joints:
        images.append(vis_joint(j))
    return np.vstack(images)
    
def train():
    with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    dataset = JointAngleDataset()
    skel_fk = SkeletonFK(mesh)
    pose_vae = PoseVae(41*3, 32)
    num_epoch = 5
    batch_size = 128

    import os, random, string
    def rand_model_name():
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))
    exp_dir = os.path.join('D:\\exp\\pose_vae\\', rand_model_name())
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    if torch.cuda.device_count() > 0:
        pose_vae = pose_vae.cuda()
        skel_fk = skel_fk.cuda()

    optimizer = torch.optim.Adam(pose_vae.parameters(), lr=1e-3)
    for epoch in range(num_epoch):
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        total_loss = 0
        for num_it, pose_para in enumerate(data_loader):
            if torch.cuda.device_count() > 0:
                pose_para = pose_para.cuda()

            joints = skel_fk(pose_para)[:,:,0:3] / 100.0
            joints = joints.view(joints.shape[0], -1)
            optimizer.zero_grad()
            recon_joints, _, _, loss = pose_vae(joints)
            loss.backward()
            total_loss += loss.item()
            if num_it %200 == 1:
                print('curr_loss: {:.5f}, avg_loss: {:.5f}'.format(loss.item(), total_loss / num_it))
                img = vis_vae_recons(joints, recon_joints)
                cv2.imwrite(os.path.join(exp_dir, '%d-%d.jpg'%(epoch, num_it)), img)

                rand_pose = pose_vae.random_sample(128)
                img = vis_vae_sample(rand_pose)
                cv2.imwrite(os.path.join(exp_dir, 'rnd_%d-%d.jpg'%(epoch, num_it)), img)
            optimizer.step()
        
        pth_path = os.path.join(exp_dir, 'model_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'network_state_dict': pose_vae.state_dict(),
        }, pth_path)

if __name__ == '__main__':
    train()