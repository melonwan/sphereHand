from __future__ import absolute_import, division, print_function
import torch

import network.constants
constant = network.constants.Constant()

def average_joint_error(gt_joints: torch.tensor, 
                        est_joints: torch.tensor):
    gt_joints = gt_joints.cpu()
    est_joints = est_joints.cpu()
    gt_joints = gt_joints[:, :, constant.real_key_points, :]
    est_joints = est_joints[:, :, constant.synt_key_points, :]
    gt_joints = gt_joints.view(-1, len(constant.real_key_points), 3)
    est_joints = est_joints.view(-1, len(constant.synt_key_points), 3)

    error = (gt_joints - est_joints).norm(dim=-1)
    return float(error.mean())
