from __future__ import absolute_import, division, print_function

import pickle
import numpy as np
from dataset.joint_angle import JointAngleDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from mesh.kinematicsTransformation import HandTransformationMat
from mesh.pointTransformation import LinearBlendSkinning, RandScale

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


def create_pose_prior():
    with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    skeleton_fk = SkeletonFK(mesh).cuda()
    batch_size = 1000
    dataset = JointAngleDataset()
    joints = []

    num_epochs = 3
    for _ in range(num_epochs):
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for _, para in enumerate(data_loader):
            joints.append(skeleton_fk(para.cuda()))
    joints = torch.cat(joints, dim=0)[:, :, 0:3]
    skel_roots = joints[:,0,:].unsqueeze(dim=1)
    joints = joints - skel_roots
    joints = joints.view(joints.shape[0], -1)

    joints =joints.detach().cpu().numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=30, svd_solver='randomized')
    pca.fit(joints)
    pca_components = pca.components_
    pca_mean = pca.mean_
    print(pca_components.shape)
    print(pca_mean.shape)

    with open('mesh/model/pose_prior-30.pkl', 'wb') as f:
        pickle.dump({'mean': pca_mean, 'components': pca_components},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_pose_prior()
