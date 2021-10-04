from __future__ import absolute_import, division, print_function
import os
import numpy as np
import torch
import torch.utils.data as data
import pickle
import os

class NpyDataset(data.Dataset):
    def __init__(self, file_path, transform=None):
        super(NpyDataset, self).__init__()
        with open(file_path + '_shape.pkl', 'rb') as f:
            shape_info  =pickle.load(f)
        file_name = file_path + '_dms.bat'
        self.dms = np.memmap(file_name, dtype='float32', mode='r', shape=shape_info['dms'])
        file_name = file_path + '_joint_poses.npy'
        self.joint_poses = np.load(file_name)
        file_name = file_path + '_camera_poses.npy'
        self.camera_poses = np.load(file_name)
        inv_camera_poses = [np.linalg.inv(m).reshape(1,4,4) for m in self.camera_poses.reshape(-1,4,4)]
        self.inv_camera_poses = np.concatenate(inv_camera_poses, axis=0).reshape(self.camera_poses.shape)
        self.transform = transform

    def __getitem__(self, index: int):
        if self.transform is None:
            return np.asarray(self.dms[index]), self.joint_poses[index], self.camera_poses[index], self.inv_camera_poses[index]
        else:
            return self.transform(np.asarray(self.dms[index]), self.joint_poses[index], self.camera_poses[index], self.inv_camera_poses[index])
    
    def __len__(self):
        return self.joint_poses.shape[0]


def create_nyu_dataset(file_dir):
    if type(file_dir) is not list:
        file_dir = [file_dir]

    datasets = []
    for dir in file_dir:
        idx = 0
        print('loading data from: ', dir)
        curr_path = os.path.join(dir, 'mv_data_%d'%idx)
        while os.path.exists(curr_path + '_shape.pkl'):
            datasets.append(NpyDataset(curr_path))
            idx += 1
            curr_path = os.path.join(dir, 'mv_data_%d'%idx)
            if os.name == 'nt' and idx > 5:
                break
    print('all_loaded')            
    return data.ConcatDataset(datasets)
    

if __name__ == '__main__':
    dataset_dir = 'D:\\data\\nyu_hand_dataset_v2\\npy-64\\train'
    nyu_dataset = create_nyu_dataset(dataset_dir)
    raise Exception

    data_loader = data.DataLoader(nyu_dataset, batch_size=64, num_workers=0, shuffle=True)

    # print('create iterator')
    # data_it = iter(data_loader)
    # print('iterator created')
    # for i in range(100):
    #     dms, joint_poses, camera_poses = next(data_it)
    #     print(i)

    for idx, (dm, pose, camera) in enumerate(data_loader):
        print(idx)
        dm = dm.numpy()
        np.save('dm.npy', dm[0,0,:,:])
        import matplotlib.pyplot as plt
        plt.imshow(dm[0,0,:,:])
        plt.show()

