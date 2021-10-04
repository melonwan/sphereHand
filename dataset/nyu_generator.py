from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import os
import scipy.io as sio
from PIL import Image
from utils import (
    crop_dm,
    CameraIntrinsic,
    estimate_rigid_transformation,
)
import pickle


class NyuDatasetGenerator():
    def __init__(self, dataset_dir, subset):
        self.cube_size = (300, 300, 300)
        self.img_size = (64, 64)
        
        self.src_dir = os.path.join(dataset_dir, 'dataset', subset)
        self.npy_dir = os.path.join(dataset_dir, 'npy-%d'%self.img_size[0], subset)
        if not os.path.exists(self.npy_dir):
            os.makedirs(self.npy_dir)
        annotation_mat = sio.loadmat(
            os.path.join(self.src_dir, 'joint_data.mat'))
        self.camera_num = 3
        self.joints = [annotation_mat['joint_xyz'][idx]
                       for idx in range(self.camera_num)]
        self.names = [['depth_{}_{:07d}.png'.format(camera_idx+1, idx+1) for idx in range(
            len(self.joints[camera_idx]))] for camera_idx in range(self.camera_num)]
        for joint in self.joints:
            joint[:, :, 1] *= -1

        self.depth_camera = CameraIntrinsic(
            fx=588.235, fy=587.084, cx=320, cy=240)
        self.render_camera = CameraIntrinsic(
            fx=self.img_size[0] / self.cube_size[0],
            fy=self.img_size[1] / self.cube_size[1],
            cx=self.img_size[0]/2,
            cy=self.img_size[1]/2)
        self.num_sample = len(self.names[0])

    def load_sample_from_file(self, idx: int):
        dms, annotations = [], []
        for camera_idx in range(self.camera_num):
            image_path = os.path.join(
                self.src_dir, self.names[camera_idx][idx])
            dm = Image.open(image_path)
            _, g, b = dm.split()
            g = np.asarray(g, np.int32)
            b = np.asarray(b, np.int32)
            dms.append(np.bitwise_or(
                np.left_shift(g, 8), b).astype(np.float32))
            annotations.append(self.joints[camera_idx][idx])
        return dms, annotations

    def crop_sample(self, dms: np.ndarray, anntations: np.ndarray):
        cropped_dms = []
        cropped_poses = []
        for dm, annotation in zip(dms, anntations):
            cropped_dm = crop_dm(
                dm, annotation[32], self.depth_camera, self.cube_size, self.img_size)
            cropped_dms.append(cropped_dm)
            cropped_poses.append(
                annotation - np.expand_dims(annotation[32], axis=0))
        cropped_dms = [np.expand_dims(d, axis=0) for d in cropped_dms]
        cropped_poses = [np.expand_dims(p, axis=0) for p in cropped_poses]
        return np.concatenate(cropped_dms), np.concatenate(cropped_poses)

    def estimate_camera_pose(self, cropped_poses):
        camera_poses = []
        for idx in range(self.camera_num):
            if idx == 0:
                camera_poses.append(np.eye(4))
            else:
                transform_mat = estimate_rigid_transformation(
                    cropped_poses[idx], cropped_poses[0])
                camera_poses.append(transform_mat)
        camera_poses = [np.expand_dims(c, axis=0) for c in camera_poses]
        return np.concatenate(camera_poses)
    

    def prepare_sample(self, idx):
        dms, annotations = self.load_sample_from_file(idx)
        cropped_dms, cropped_poses = self.crop_sample(dms, annotations)
        camera_poses = self.estimate_camera_pose(cropped_poses)
        return cropped_dms, cropped_poses, camera_poses

    def create_npy_dataset_from_indices(self, file_name: str, indices: list):
        dms, joint_poses, camera_poses = [], [], []
        for idx in indices:
            dm, joint_pose, camera_pose = self.prepare_sample(idx)
            dms += [np.expand_dims(dm, axis=0)]
            joint_poses += [np.expand_dims(joint_pose, axis=0)]
            camera_poses += [np.expand_dims(camera_pose, axis=0)]
            if idx % 500 == 0:
                print('%d samples has been loaded'%idx)
        
        dms = np.concatenate(dms).astype(np.float32)
        joint_poses = np.concatenate(joint_poses).astype(np.float32)
        camera_poses = np.concatenate(camera_poses).astype(np.float32)

        print('begin writting to the file')
        file_path = os.path.join(self.npy_dir, file_name+'_shape.pkl')
        shape_info = {}
        shape_info['dms'] = dms.shape
        shape_info['joint_poses'] = joint_poses.shape
        shape_info['camera_poses'] = camera_poses.shape
        with open(file_path, 'wb') as f:
            pickle.dump(shape_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_path = os.path.join(self.npy_dir, file_name+'_dms.bat')
        fp = np.memmap(file_path, dtype='float32', mode='w+', shape=dms.shape)
        fp[:] = dms[:]
        fp.flush()
        file_path = os.path.join(self.npy_dir, file_name+'_joint_poses.npy')
        np.save(file_path, joint_poses)
        file_path = os.path.join(self.npy_dir, file_name+'_camera_poses.npy')
        np.save(file_path, camera_poses)
        print('npy file has been written to %s'%file_path)
        
    
    def create_npy_dataset(self, num_samples_per_segment: int):
        num_files = self.num_sample // num_samples_per_segment + 1
        for idx in range(num_files):
            start = idx*num_samples_per_segment
            end = min(start + num_samples_per_segment, self.num_sample)
            file_indices = list(range(start, end))
            file_name = 'mv_data_%d'%idx
            print('create files from %d to %d'%(start, end))
            self.create_npy_dataset_from_indices(file_name, file_indices)

parser = argparse.ArgumentParser()
parser.add_argument('--nyu_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    dataset_generator = NyuDatasetGenerator(args.nyu_path, 'train')
    dataset_generator.create_npy_dataset(1000)

    dataset_generator = NyuDatasetGenerator(args.nyu_path, 'test')
    dataset_generator.create_npy_dataset(1000)
