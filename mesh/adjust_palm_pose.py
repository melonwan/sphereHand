from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.linalg import svd, det
import math
import random


palm_joint_indices = list(range(11))
canonical_palm_pts = np.asarray([
    [-15.406372,  79.6443, 52.62097],
    [15.9734955, 82.24512, 47.8826],
    [ 2.219452, 13.717743, 26.17221],
    [-20.76764,   23.576614,  22.998213],
    [ 1.736557, 36.935516, 35.718056],
    [ 0.74661255, 84.52956,    55.28605],
    [ 1.1482239, 65.221634,  47.347717],
    [-10.020676,  66.30975,   44.136154],
    [-14.763321,  46.575455,  36.9746],
    [17.22052,  23.905838, 24.11928],
    [16.956161, 40.65445,  31.56168],
    [ 7.968979, 68.520035, 21.761257]
])

template_indices_for_estimation = [
    2, 4, 8, 10
]
real_indices_for_estimation = [
    2, 4, 8, 10
]

template_indices_for_replace = list(range(11))

def similarity_transformation(pt, R, t, scale):
    t = t.reshape((1, 3))
    return scale * np.matmul(pt, R.T) + t

def estimate_similarity_transformation(
    pt1: np.ndarray, pt2: np.ndarray, do_scale: bool = True):
    if len(pt1) < 3:
        return np.eye(3), np.zeros((3, 1)), 1.0

    c1 = pt1.mean(axis=0)
    c2 = pt2.mean(axis=0)

    pt1 = pt1 - np.expand_dims(c1, axis=0)
    pt2 = pt2 - np.expand_dims(c2, axis=0)
    
    s1 = np.mean(np.sqrt(np.sum(pt1**2, axis=1)))
    s2 = np.mean(np.sqrt(np.sum(pt2**2, axis=1)))

    if do_scale:
        scale = s1 / s2
    else:
        scale = 1.0
    
    pt1 = pt1 / s1 * math.sqrt(3.0)
    pt2 = pt2 / s2 * math.sqrt(3.0)

    H = np.matmul(pt1.T, pt2)
    U, S, V = svd(H)
    S = np.asarray([[1,0,0], [0,1,0], [0,0,det(np.matmul(V, U.T))]])
    R = np.matmul(np.matmul(V, S), U.T)

    t = -scale*np.matmul(R, c1) + c2
    return R, t, scale

def estimate_similarity_transformation_ransac(
    pt1: np.ndarray, pt2: np.ndarray, do_scale: bool=True, iteration: int=10, inlier_thresh: float=15.0
): 
    num_pts = len(pt1)
    opt_inliers = []
    for _ in range(iteration):
        rnd_indices = list(range(num_pts))
        random.shuffle(rnd_indices)
        rnd_indices = rnd_indices[:3]
        R, t, scale = estimate_similarity_transformation(pt1[rnd_indices], pt2[rnd_indices])
        rse = np.sqrt(((similarity_transformation(pt1, R, t, scale) - pt2)**2).sum(axis=-1))
        inlier = [idx for idx, e in enumerate(rse) if e < inlier_thresh]
        if len(opt_inliers) < len(inlier):
            opt_inliers = inlier
    return estimate_similarity_transformation(pt1[opt_inliers], pt2[opt_inliers])






def adjust_palm_pose(joints: np.ndarray):
    joints = joints.reshape(-1, 3)
    # indices of palm starts from zero
    R, t, scale = estimate_similarity_transformation(
        canonical_palm_pts[template_indices_for_estimation], joints[real_indices_for_estimation], True)
    transformed_palm_pts = similarity_transformation(canonical_palm_pts[template_indices_for_replace], R, t, scale)
    adjusted_pts = joints.copy()
    adjusted_pts[palm_joint_indices] = transformed_palm_pts
    return adjusted_pts