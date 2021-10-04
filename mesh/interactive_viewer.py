from __future__ import print_function, absolute_import, division

import pickle
import torch
from kinematicsTransformation import HandTransformationMat
from pointTransformation import LinearBlendSkinning, OthographicalProjection
from render import HandBallPrimitiveRender, DepthRender
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

if __name__ == '__main__':
    with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)

    vertices = mesh['vertices']
    offset_mats = []
    skinning_weights = []
    skinning_vertex_indices = []
    for bone in mesh['bones']:
        offset_mats.append(bone['offset_matrix'].astype(np.float32))
        skinning_vertex_indices.append(bone['weight_vertexid'])
        skinning_weights.append(bone['weight_coeff'])
    hand_skeleton_transform = HandTransformationMat(offset_mats)
    lbs = LinearBlendSkinning(
        vertices, skinning_weights, skinning_vertex_indices)
    camera = OthographicalProjection(320, 320, 640/300, 640/300)

    primitive_render = HandBallPrimitiveRender(mesh['bones'], 64, 64)
    depth_render = DepthRender(mesh, 64)

    if torch.cuda.device_count() > 0:
        hand_skeleton_transform = hand_skeleton_transform.cuda()
        primitive_render = primitive_render.cuda()
        depth_render = depth_render.cuda()
        camera = camera.cuda()
        lbs = lbs.cuda()

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.6, bottom=0.1, top = 0.9, right = 0.9)
    sliders = []
    for idx in range(30):
        ax = plt.axes([0.0,  0.0 + 0.03*idx, 0.5, 0.02])
        if idx >=3 and idx < 6:
            s = Slider(ax, 'palm translation', -20, 20)
        elif idx < 3:
            s = Slider(ax, 'palm rotation', -pi, pi)
        else:
            s = Slider(ax, '', -1.8, 1.8, 0)
        sliders.append(s)

    ax = fig.add_subplot(2, 1, 1)
    primitive_ax = ax.imshow(np.zeros((64, 64),np.float32), vmin=-100.0, vmax=100.0, cmap='gray')
    ax = fig.add_subplot(2, 1, 2)
    rastered_ax = ax.imshow(np.zeros((64, 64),np.float32), vmin=-100.0, vmax=100.0, cmap='gray')

    def render(parameters):
        parameters = torch.from_numpy(parameters).type(torch.float).cuda()
        transformed_mat = hand_skeleton_transform(parameters)
        _, primitive_dms = primitive_render(transformed_mat)
        rastered_dms = depth_render(transformed_mat)

        rastered_dms = rastered_dms.cpu().numpy()
        primitive_dms = primitive_dms.cpu().numpy()
        return rastered_dms[0,:,:], primitive_dms[0,:,:]

    def update(val):
        parameters = np.zeros((1,30), np.float32)
        for idx, s in enumerate(sliders):
            parameters[0,idx] = s.val
        rastered_dm, primitive_dm = render(parameters)
        primitive_ax.set_array(primitive_dm)
        rastered_ax.set_array(rastered_dm)
        fig.canvas.draw()

    for s in sliders:
        s.on_changed(update)

    plt.show()
    

