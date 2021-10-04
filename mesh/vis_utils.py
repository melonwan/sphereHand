from __future__ import print_function, division, absolute_import
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

def visualize_vertices(vertices:np.ndarray, bones:np.ndarray = None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(vertices[:-1:5,0], vertices[:-1:5,1], vertices[:-1:5,2], c='b')

    print('%f to %f' % (vertices[:,2].min(), vertices[:,2].max()))

    if bones is not None:
        joints = []
        for bone in bones:
            joint = np.linalg.inv(bone['offset_matrix'])[0:3, 3]
            joints.append(np.expand_dims(joint, axis=0))
        joints = np.vstack(joints)
        ax.scatter(joints[:,0], joints[:,1], joints[:,2], c='r')
        print('%f to %f' % (joints[:,2].min(), joints[:,2].max()))


        
    plt.show()

if __name__ == '__main__':
    with open('mesh/model/preprocessed_right_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    visualize_vertices(mesh['vertices'], mesh['bones'])