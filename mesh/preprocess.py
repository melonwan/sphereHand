from __future__ import print_function, absolute_import, division
import pickle
import numpy as np

_scale_ratio = float(58.0)


def scale_mesh(mesh: dict):
    mesh['vertices'] *= _scale_ratio
    mesh['vertices'][:,2] *= -1

    def scale_offset_mat(offset: np.ndarray) -> np.ndarray:
        transform = np.linalg.inv(offset)
        transform[0][3] *= _scale_ratio
        transform[1][3] *= _scale_ratio
        transform[2][3] *= -_scale_ratio
        return np.linalg.inv(transform)

    for bone in mesh['bones']:
        bone['offset_matrix'] = scale_offset_mat(bone['offset_matrix'])
    return mesh


def center_mesh(mesh: dict):
    center = np.mean(mesh['vertices'], axis=0)
    mesh['vertices'] -= center

    def translate_offset_mat(offset: np.ndarray) -> np.ndarray:
        transform = np.linalg.inv(offset)
        transform[0:3, 3] -= center
        return np.linalg.inv(transform)

    for bone in mesh['bones']:
        bone['offset_matrix'] = translate_offset_mat(bone['offset_matrix'])
    return mesh


def vertices_to_homography(vertices):
    num_vertices = vertices.shape[0]
    homography_val = np.ones((num_vertices, 1))
    vertices = np.hstack([vertices, homography_val])
    return vertices


def delete_palm_bones(bones: list):
    bones[0], bones[1] = bones[1], bones[0]
    root = bones[0]
    root_weight_map = {}
    for weight, index in zip(root['weight_coeff'], root['weight_vertexid']):
        root_weight_map[index] = weight
    for _ in range(4):
        bone = bones.pop(-1)
        for weight, index in zip(bone['weight_coeff'], bone['weight_vertexid']):
            if index in root_weight_map:
                root_weight_map[index] += weight
            else:
                root_weight_map[index] = weight

    weight_vertexid, weight_coeff = [], []
    for index, weight in root_weight_map.items():
        weight_vertexid.append(index)
        weight_coeff.append(weight)

    root['weight_coeff'] = np.asarray(weight_coeff, np.float)
    root['weight_vertexid'] = np.asarray(weight_vertexid, np.int64)
    return bones


def add_keypoint(bones: list):
    offsets = [  # the local coordinate of the corresponding joint
        [-0.1355, -0.00849999, -0.2875],  # F1_KNU3_A,
        [0.002, 0.007, -0.1205],  # F1_KNU3_B,
        [-0.13, 0.0305, -0.1975],  # F1_KNU2_A,
        [0.0295, 0.00149996, -0.0615],  # F1_KNU2_B,
        [-0.3195, 0.0315, -0.211],  # F1_KNU1_A,
        [0.0115, -0.0235, -0.1275],  # F1_KNU1_B,
        [-0.2615, -0.1135, -0.3965],  # F2_KNU3_A,
        [-0.126, -0.0245, -0.131],  # F2_KNU3_B,
        [-0.144, -0.00450001, -0.0855],  # F2_KNU2_A,
        [0.0705, 0.00400001, 0.03],  # F2_KNU2_B,
        [-0.3505, -0.0275, -0.281],  # F2_KNU1_A,
        [-0.002, -0.0635, -0.1945],  # F2_KNU1_B,
        [-0.157, -0.0285, -0.279],  # F3_KNU3_A,
        [-0.0195, 0.0375, 0.001],  # F3_KNU3_B,
        [-0.1665, 0.022, -0.205],  # F3_KNU2_A,
        [0.029, 0.0545, -0.0535],  # F3_KNU2_B,
        [-0.419, 0.0565, -0.044],  # F3_KNU1_A,
        [-0.0095, 0.0005, 0.0085],  # F3_KNU1_B,
        [-0.343, 0.012, -0.3445],  # F4_KNU3_A,
        [-0.144, 0.0295, -0.189],  # F4_KNU3_B,
        [-0.2485, 0.008, -0.172],  # F4_KNU2_A,
        [0.0, 0.0335, -0.0125],  # F4_KNU2_B,
        [-0.5595, -0.035, -0.0315],  # F4_KNU1_A,
        [-0.0325, -0.0405, 0.0],  # F4_KNU1_B,
        [-0.432, 0.0775, -0.104],  # TH_KNU3_A,
        [-0.066, 0.0950001, -0.038],  # TH_KNU3_B,
        [-0.341, 0.017, 0.0175],  # TH_KNU2_A,
        [-0.0335, 0.0585, 0.044],  # TH_KNU2_B,
        [-0.4485, -0.343, -0.115],  # TH_KNU1_A,
        [0.0, 0.0, 0.0],  # TH_KNU1_B,
        [-0.1, 0.305, -0.064],  # PALM_1,
        [-0.1, -0.305, -0.064],  # PALM_2,
        [-1.467, 0.0, 0.0],  # PALM_3,
        [-1.307, 0.4095, -0.2],  # PALM_4,
        [-0.986, 0.0, 0.0],  # PALM_5,
        [0.0, 0.0, 0.0],  # PALM_6
        [-0.4, 0.0, 0.0],
        [-0.4, 0.20, -0.1],
        [-0.8, 0.30, -0.1],
        [-1.307, -0.305, -0.06],
        [-0.956, -0.305, -0.05]]
    bone_indices = [  # corresponding bones for each
        'finger1joint3',  # F1_KNU3_A,
        'finger1joint3',  # F1_KNU3_B,
        'finger1joint2',  # F1_KNU2_A,
        'finger1joint2',  # F1_KNU2_B,
        'finger1joint1',  # F1_KNU1_A,
        'finger1joint1',  # F1_KNU1_B,
        'finger2joint3',  # F2_KNU3_A,
        'finger2joint3',  # F2_KNU3_B,
        'finger2joint2',  # F2_KNU2_A,
        'finger2joint2',  # F2_KNU2_B,
        'finger2joint1',  # F2_KNU1_A,
        'finger2joint1',  # F2_KNU1_B,
        'finger3joint3',  # F3_KNU3_A,
        'finger3joint3',  # F3_KNU3_B,
        'finger3joint2',  # F3_KNU2_A,
        'finger3joint2',  # F3_KNU2_B,
        'finger3joint1',  # F3_KNU1_A,
        'finger3joint1',  # F3_KNU1_B,
        'finger4joint3',  # F4_KNU3_A,
        'finger4joint3',  # F4_KNU3_B,
        'finger4joint2',  # F4_KNU2_A,
        'finger4joint2',  # F4_KNU2_B,
        'finger4joint1',  # F4_KNU1_A,
        'finger4joint1',  # F4_KNU1_B,
        'finger5joint3',  # TH_KNU3_A,
        'finger5joint3',  # TH_KNU3_B,
        'finger5joint2',  # TH_KNU2_A,
        'finger5joint2',  # TH_KNU2_B,
        'finger5joint1',  # TH_KNU1_A,
        'finger5joint1',  # TH_KNU1_B,
        'metacarpals',  # PALM_1,
        'metacarpals',  # PALM_2,
        'metacarpals',  # PALM_3,
        'metacarpals',  # PALM_4,
        'metacarpals',  # PALM_5,
        'metacarpals',  # PALM_6,
        'metacarpals',  # ,
        'metacarpals',  # ,
        'metacarpals',  # ,
        'metacarpals',  # ,
        'metacarpals',  # ,
    ]
    radiuses = [
        0.100,  # F1_KNU3_A,
        0.120,   # F1_KNU3_B,
        0.140,   # F1_KNU2_A,
        0.160,   # F1_KNU2_B,
        0.170,   # F1_KNU1_A,
        0.210,   # F1_KNU1_B,
        0.120,   # F2_KNU3_A,
        0.140,   # F2_KNU3_B,
        0.170,   # F2_KNU2_A,
        0.180,   # F2_KNU2_B,
        0.190,   # F2_KNU1_A,
        0.200,   # F2_KNU1_B,
        0.150,   # F3_KNU3_A,
        0.170,   # F3_KNU3_B,
        0.180,   # F3_KNU2_A,
        0.200,   # F3_KNU2_B,
        0.200,   # F3_KNU1_A,
        0.210,   # F3_KNU1_B,
        0.140,   # F4_KNU3_A,
        0.160,   # F4_KNU3_B,
        0.170,   # F4_KNU2_A,
        0.180,   # F4_KNU2_B,
        0.200,   # F4_KNU1_A,
        0.210,   # F4_KNU1_B,
        0.170,   # TH_KNU3_A,
        0.190,   # TH_KNU3_B,
        0.200,   # TH_KNU2_A,
        0.250,   # TH_KNU2_B,
        0.280,   # TH_KNU1_A,
        0.280,   # TH_KNU1_B,
        0.250,   # PALM_1
        0.250,   # PALM_2
        0.280,   # PALM_3
        0.280,   # PALM_4
        0.300,   # PALM_5
        0.280,   # PALM_6
        0.250,
        0.250,
        0.250,
        0.250,
        0.250
    ]

    bone_name_dict = {}
    for idx, bone in enumerate(bones):
        bone_name_dict[bone['name']] = idx

    for offset, bone_name, radius in zip(offsets, bone_indices, radiuses):
        bone = bones[bone_name_dict[bone_name]]
        if 'keypoint' not in bone:
            bone['keypoint'] = []
        transform_mat = np.linalg.inv(bone['offset_matrix'])
        offset = np.asarray(offset) * _scale_ratio
        offset[2] *= -1.0
        keypoint = transform_mat[0:3, 3] + offset
        bone['keypoint'].append((keypoint, radius * 80.0))
        # bone['keypoint'].append((keypoint, radius * 58.0))
        # bone['keypoint'].append((keypoint, radius * 100.0))
    return bones


with open('mesh/model/hand.pkl', 'rb') as f:
    mesh = pickle.load(f, encoding='latin1')

mesh = center_mesh(scale_mesh(mesh))
mesh['bones'] = delete_palm_bones(mesh['bones'])
mesh['bones'] = add_keypoint(mesh['bones'])
mesh['vertices'] = vertices_to_homography(mesh['vertices'])

with open('mesh/model/preprocessed_hand.pkl', 'wb') as f:
    pickle.dump(mesh, f, protocol=pickle.HIGHEST_PROTOCOL)
