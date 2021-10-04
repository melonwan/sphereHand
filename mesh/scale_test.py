# to test the gradient back-propagation
from __future__ import absolute_import, division, print_function
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.utils.data as data
from mesh.multiview_utility import MutualProjectionLoss, MultiviewConsistencyLoss
from dataset.nyu_dataset import create_nyu_dataset
from network.util_modules import PosePriorLoss
from dataset.joint_angle import JointAngleDataset
import matplotlib.pyplot as plt

from mesh.kinematicsTransformation import HandTransformationMat
from mesh.pointTransformation import LinearBlendSkinning, RandScale
from network.constants import Constant
from network.util_modules import HandSynthesizer
constant = Constant()

synt_key_points = [[0,1],
                   [0,2],
                   [33,34]]

real_key_points = [[30,31],
                   [30,32],
                   [0,1]]

def bone_length(joints, indices):
    bone_length = []
    for [idx_1, idx_2] in indices:
        diff = joints[idx_1] - joints[idx_2]
        bone_length.append(diff.view(-1).norm())
    bone_length = [bone_length[0]/bone_length[1], bone_length[0]/bone_length[2]]
    return bone_length



dataset_dir = 'D:\\data\\nyu_hand_dataset_v2\\npy-64\\test'
nyu_dataset = create_nyu_dataset(dataset_dir)

with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
    mesh = pickle.load(f)
hand_synthsizer = HandSynthesizer(mesh, 64, 16, 1.0, 0.01).cuda()
joint_data = JointAngleDataset()


for _ in range(10):
    real_data = nyu_dataset[0]
    real_joints = real_data[1][0][constant.real_key_points]
    real_joints = real_joints * 64 / 300 + 32


    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(real_joints[:,0], real_joints[:,1])
    for idx in range(len(real_joints)):
        ax.annotate('%d'%idx, (real_joints[idx,0], real_joints[idx,1]))
    ax.imshow(real_data[0][0].squeeze())


    para = joint_data[0].unsqueeze(dim=0).cuda()

    synt_result = hand_synthsizer(para)
    dm = synt_result[0].squeeze().detach().cpu().numpy()
    synt_joints = synt_result[3].squeeze().detach().cpu().numpy()[constant.synt_key_points]
    synt_joints = synt_joints * 64 / 300 + 32

    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(synt_joints[:,0], synt_joints[:,1])
    for idx in range(len(synt_joints)):
        ax.annotate('%d'%idx, (synt_joints[idx,0], synt_joints[idx,1]))
    ax.imshow(dm)
    plt.show()