from __future__ import absolute_import, division, print_function
import pickle

with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
    _mesh = pickle.load(f)

with open('mesh/model/pose_prior.pkl', 'rb') as f:
    _pca = pickle.load(f)

class Constant:
    depthmap_size = 64
    heatmap_size = 16
    num_joint = 41
    depth_scale = 1.0 / 100.0
    uv_hm_scale = 1.0
    mesh = _mesh
    joint_color = [(255, 0, 0)] * 11 +\
                  [(25, 255, 25)]*6 +\
                  [(212, 0, 255)]*6 +\
                  [(0, 230, 230)]*6 +\
                  [(179, 179, 0)]*6 +\
                  [(255, 153, 153)]*6

    gt_joint_color = [(179, 179, 0)]*6 +\
                    [(0, 230, 230)]*6 +\
                    [(212, 0, 255)]*6 +\
                    [(25, 255, 25)]*6 +\
                    [(255, 153, 153)]*6+\
                    [(255, 0, 0)] * 11
    synt_key_points = [33,32, 27,26, 21,20, 15,14, 39,40,38, 0,1,2]
    real_key_points = [0,3, 6,9, 12,15, 18,21, 24,25,27, 30,31,32]

    pca_component = _pca['components']
    pca_mean = _pca['mean']