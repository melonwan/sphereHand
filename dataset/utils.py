from __future__ import absolute_import, division, print_function
import numpy as np
from typing import NamedTuple
import matplotlib.pyplot as plt


class CameraIntrinsic(NamedTuple):
    fx: float = 588.235
    fy: float = 587.084
    cx: int = 320
    cy: int = 240


def perspective_projection(xyz_point, camera):
    if xyz_point.ndim == 1:
        uvd_point = np.zeros((3))
        uvd_point[0] = xyz_point[0] * camera.fx / xyz_point[2] + camera.cx
        uvd_point[1] = xyz_point[1] * camera.fy / xyz_point[2] + camera.cy
        uvd_point[2] = xyz_point[2]
    elif xyz_point.ndim == 2:
        num_point = xyz_point.shape[0]
        uvd_point = np.zeros((num_point, 3))
        uvd_point[:, 0] = xyz_point[:, 0] * \
            camera.fx / xyz_point[:, 2] + camera.cx
        uvd_point[:, 1] = xyz_point[:, 1] * \
            camera.fy / xyz_point[:, 2] + camera.cy
        uvd_point[:, 2] = xyz_point[:, 2]
    else:
        raise ValueError('unknown input point shape')

    return uvd_point


def perspective_back_projection(uvd_point, camera):
    if uvd_point.ndim == 1:
        xyz_point = np.zeros((3))
        xyz_point[0] = (uvd_point[0] - camera.cx) * uvd_point[2] / camera.fx
        xyz_point[1] = (uvd_point[1] - camera.cy) * uvd_point[2] / camera.fy
        xyz_point[2] = uvd_point[2]
    elif uvd_point.ndim == 2:
        num_point = uvd_point.shape[0]
        xyz_point = np.zeros((num_point, 3))
        xyz_point[:, 0] = (uvd_point[:, 0] - camera.cx) * \
            uvd_point[:, 2] / camera.fx
        xyz_point[:, 1] = (uvd_point[:, 1] - camera.cy) * \
            uvd_point[:, 2] / camera.fy
        xyz_point[:, 2] = uvd_point[:, 2]
    else:
        raise ValueError('unknown input point shape')
    return xyz_point


def othorgraphical_projection(xyz_point, camera):
    if xyz_point.ndim == 1:
        uvd_point = np.zeros((3))
        uvd_point[0] = xyz_point[0] * camera.fx + camera.cx
        uvd_point[1] = xyz_point[1] * camera.fy + camera.cy
        uvd_point[2] = xyz_point[2]
    elif xyz_point.ndim == 2:
        num_point = xyz_point.shape[0]
        uvd_point = np.zeros((num_point, 3))
        uvd_point[:, 0] = xyz_point[:, 0] * camera.fx + camera.cx
        uvd_point[:, 1] = xyz_point[:, 1] * camera.fy + camera.cy
        uvd_point[:, 2] = xyz_point[:, 2]
    else:
        raise ValueError('unknown input point shape')
    return uvd_point


def crop_dm(dm: np.ndarray,
            xyz_center: np.ndarray,
            depth_camera: CameraIntrinsic,
            cube_size: tuple,
            img_size: tuple,
            far_point_value: float = 100.0) -> np.ndarray:
    assert dm.ndim == 2, 'unknown dimension of depth map, should be 2'
    dm_height, dm_width = dm.shape[0], dm.shape[1]

    uvd_center = perspective_projection(xyz_center, depth_camera)
    assert uvd_center[0] >= 0 and uvd_center[0] < dm_width
    assert uvd_center[1] >= 0 and uvd_center[1] < dm_height

    z_start = float(xyz_center[2] - cube_size[2] / 2)
    z_end = float(xyz_center[2] + cube_size[2] / 2)

    top_left = perspective_projection(xyz_center + np.asarray([-cube_size[0]/2, -cube_size[1]/2, -cube_size[2]/2], dtype=np.float32),
                                      depth_camera)
    bottom_right = perspective_projection(xyz_center + np.asarray([cube_size[0]/2, cube_size[1]/2, -cube_size[2]/2], dtype=np.float32),
                                          depth_camera)

    u_start = int(max(top_left[0], 0))
    u_end = int(min(bottom_right[0], dm_width))
    v_start = int(max(top_left[1], 0))
    v_end = int(min(bottom_right[1], dm_height))

    cropped_dm = np.ones(img_size) * far_point_value
    render_camera = CameraIntrinsic(
        fx=img_size[0] / cube_size[0], fy=img_size[1] / cube_size[1], cx=img_size[0]/2, cy=img_size[1]/2)

    dm_roi = dm[v_start:v_end, u_start:u_end]
    mask = np.logical_and(dm_roi >= z_start, dm_roi < z_end)
    u_coord, v_coord = np.meshgrid(
        range(u_start, u_end), range(v_start, v_end))
    u_coord = u_coord[mask].astype(np.float32).reshape((-1, 1))
    v_coord = v_coord[mask].astype(np.float32).reshape((-1, 1))
    d_coord = dm_roi[mask].reshape((-1, 1))

    uvd = np.concatenate([u_coord, v_coord, d_coord], axis=1)
    uvd = othorgraphical_projection(
        perspective_back_projection(
            uvd, depth_camera) - xyz_center.reshape((1, 3)),
        render_camera)
    u_coor = uvd[:, 0].astype(np.int32)
    v_coor = uvd[:, 1].astype(np.int32)

    mask_u = np.logical_and(u_coor >= 0, u_coor < img_size[0])
    mask_v = np.logical_and(v_coor >= 0, v_coor < img_size[1])
    mask = np.logical_and(mask_u, mask_v)

    u_coor = u_coor[mask]
    v_coor = v_coor[mask]
    d = uvd[:, 2][mask]
    cropped_dm[v_coor, u_coor] = d
    return cropped_dm


def estimate_rigid_transformation(point_1: np.ndarray, point_2: np.ndarray):
    assert point_1.ndim == 2 and point_1.shape[1] == 3
    assert point_2.ndim == 2 and point_2.shape[1] == 3

    center_1 = point_1.mean(axis=0)
    center_2 = point_2.mean(axis=0)
    centered_point_1 = point_1 - np.expand_dims(center_1, axis=0)
    centered_point_2 = point_2 - np.expand_dims(center_2, axis=0)
    H = np.matmul(centered_point_1.T, centered_point_2)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.matmul(Vt.T, U.T)
    t = np.matmul(-R, center_1.reshape(3, 1)) + center_2.reshape((3, 1))
    transform_mat = np.eye(4)
    transform_mat[:3, :3] = R
    transform_mat[3, :3] = t.reshape(3)
    return transform_mat


def rigid_transformation(point, transform_mat):
    if point.ndim == 1:
        point = np.expand_dims(point, axis=0)
    if point.shape[1] == 4:
        return np.matmul(point.T, transform_mat.T).T

    return np.matmul(point, transform_mat[:3, :3].T) + np.expand_dims(transform_mat[3, :3], axis=0)

def visualize_sample(cropped_dms: np.ndarray, 
                     cropped_poses: np.ndarray, 
                     camera_poses: np.ndarray,
                     render_camera: CameraIntrinsic):
        camera_num = len(cropped_dms)
        c = ['r', 'b', 'g']
        fig = plt.figure()
        for idx in range(camera_num):
            ax = fig.add_subplot(1,camera_num,1+idx)
            ax.imshow(cropped_dms[idx])
            for sub_idx in range(camera_num):
                camera_transform = np.matmul(np.linalg.inv(camera_poses[idx]), camera_poses[sub_idx]) 
                canonical_pose = rigid_transformation(cropped_poses[sub_idx], camera_transform)
                x, y = [], []
                for pt in canonical_pose:
                    pt = othorgraphical_projection(pt, render_camera)
                    x.append(pt[0])
                    y.append(pt[1])
                ax.scatter(x, y, c = c[sub_idx])
        plt.show()