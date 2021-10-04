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

def cal_bone_length(joints, indices_1, indices_2):
    bone_length = []
    for idx_1, idx_2 in zip(indices_1, indices_2):
        diff = joints[idx_1] - joints[idx_2]
        bone_length.append(diff.view(-1).norm().item())
    return bone_length


with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
    mesh = pickle.load(f)
hand_synthsizer = HandSynthesizer(mesh, 64, 16, 1.0, 0.01).cuda()
joint_data = JointAngleDataset()

joint_1, joint_2 = [], []

joint_1 += [3, 2, 3, 8, 2, 2, 9]
joint_2 += [2, 9, 8, 2, 4, 10, 10]

joint_1 += [8, 4, 8, 7, 4, 6]
joint_2 += [4, 10, 7, 4, 6, 10]

joint_1 += [7, 0, 5, 7, 7, 6, 6]
joint_2 += [6, 5, 1, 0, 5, 5, 1]

# for the finger
for idx in range(5):
    joint_1.append(11+idx*6)
    joint_2.append(12+idx*6)
    joint_1.append(13+idx*6)
    joint_2.append(14+idx*6)
    joint_1.append(15+idx*6)
    joint_2.append(16+idx*6)

uniform_length = [25.212656021118164, 18.249488830566406, 27.5742244720459, 38.532264709472656, 25.10819435119629, 31.173757553100586, 18.329626083374023, 19.15080451965332, 16.209327697753906, 21.52261734008789, 32.740535736083984, 30.58920669555664, 33.205970764160156, 11.672294616699219, 17.084707260131836, 17.084720611572266, 16.697546005249023, 23.92103385925293, 20.87999725341797, 22.58038330078125, 27.55999755859375, 15.471183776855469, 13.214692115783691, 21.748210906982422, 13.021653175354004, 16.643720626831055, 18.83765983581543, 12.724685668945312, 16.238431930541992, 18.04928970336914, 11.045844078063965, 11.320968627929688, 30.078536987304688, 16.255985260009766, 19.434825897216797]