from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import cv2
from network.constants import Constant
constant = Constant()

def vis_heatmap(dm: np.ndarray, hms: np.ndarray, colors: list):
    dm = dm.copy().astype(np.float)
    for hm, color in zip(hms, colors):
        color = np.asarray(color).reshape((1, 1, 3))
        dm = hm * color + (1.0-hm) * dm
    dm = np.clip(dm ,0, 255).astype(np.uint8)
    return dm


def vis_joints(dm: np.ndarray, joints: np.ndarray, colors: list):
    dm = dm.copy()
    for j, c in zip(joints, colors):
        cv2.circle(dm, (j[0], j[1]), 3, c, -1)
    return dm

def vis_depthmap(dm: np.ndarray):
    dm = dm.squeeze()
    dm = np.clip(dm, -1.0, 1.0)
    dm = ((dm+1.0) * 127).astype(np.uint8)
    dm = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)
    return dm

def vis_result(dms: torch.tensor,
               hms: torch.tensor, 
               joints: torch.tensor,
               vis_indices: list=None,
               output_size: tuple=(128, 128),
               resized_dms: torch.tensor = None,
               segmented_dms: torch.tensor = None):
    dms = dms.unsqueeze(dim=1)
    dms = torch.nn.functional.interpolate(dms, output_size, mode='bilinear', align_corners=False)
    dms = dms.squeeze()

    if resized_dms is not None:
        resized_dms = resized_dms.unsqueeze(dim=1)
        resized_dms = torch.nn.functional.interpolate(resized_dms, output_size, mode='bilinear', align_corners=False)
        resized_dms = resized_dms.squeeze()
        resized_dms = resized_dms.detach().cpu().numpy()

    hms = torch.nn.functional.interpolate(hms, output_size, mode='bilinear', align_corners=False)
    hms = hms.unsqueeze(dim=-1).repeat(1, 1, 1, 1, 3)

    dms = dms.detach().cpu().numpy()
    hms = hms.detach().cpu().numpy()
    joints = joints.detach().cpu().numpy()
    joint_color = constant.joint_color
    
    joints[:,:,0] = joints[:,:,0] * output_size[0] / 300 + output_size[0] / 2
    joints[:,:,1] = joints[:,:,1] * output_size[1] / 300 + output_size[1] / 2

    if vis_indices is not None:
        hms = hms[:,vis_indices,:,:]
        joints = joints[:,vis_indices,:]
        joint_color = [joint_color[idx] for idx in vis_indices]

    images = []
    for idx, (dm, hm, joint) in enumerate(zip(dms, hms, joints)):
        vis_dm = vis_depthmap(dm)
        if resized_dms is not None:
            vis_resized_dms = vis_depthmap(resized_dms[idx])
            vis_hm = vis_heatmap(vis_resized_dms, hm, joint_color)
        else:
            vis_hm = vis_heatmap(vis_dm, hm, joint_color)
        vis_skel = vis_joints(vis_dm, joint, joint_color)
        img = np.hstack([vis_dm, vis_hm, vis_skel])
        images.append(img)
    return np.vstack(images)