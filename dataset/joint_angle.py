from __future__ import absolute_import, division, print_function
import torch
import torch.utils.data as data
from math import pi


class JointAngleDataset(data.Dataset):
    INDEX = 6
    MIDDLE = 10
    RING = 14
    PINKY = 18
    THUMB = 22
    ABDUCT = 0
    FLEX_1 = 1
    FLEX_2 = 2
    FLEX_3 = 3
    
    def __init__(self):
        super(JointAngleDataset, self).__init__()
        self.num_parameter = 26
    
    def _set_palm(self):
        x_angle = torch.rand(1)*6.28 - 3.14
        y_angle = -torch.rand(1)*3.14
        z_angle = torch.rand(1)*6.28 - 3.14
        x_trans = torch.rand(1)*30 - 15
        y_trans = torch.rand(1)*30 - 15
        z_trans = torch.rand(1)*50 - 35
        return torch.tensor([x_angle, y_angle, z_angle, x_trans, y_trans, z_trans])
        # return torch.tensor([2.98, -0.39, 1.55, 0, 0, 0]).type(torch.float)

    def _set_abduct(self):
        spread = (torch.rand(1) - 0.35) / 1.55
        def rand_abudct():
            return (torch.rand(1)*10 - 5) * pi / 180 
        index_abudct = 1.55 * (spread + rand_abudct())
        middle_abudct = 0.75 * (spread + rand_abudct())
        ring_abduct = - 0.75 * (spread + rand_abudct())
        pinky_abduct = - 2.2 * (spread + rand_abudct())
        return [index_abudct, middle_abudct, ring_abduct, pinky_abduct]
    
    def _set_closed_finger(self):
        def pertubation():
            return (torch.rand(1)*20 - 10) * pi / 180 
        def rand_flex():
            return (torch.rand(1) * 30 + 60) * pi / 180

        flex_1, flex_2, flex_3 = -0.2, -0.4, -0.34
        curr_flex = rand_flex() + pertubation()
        flex_1 += 1.0 * curr_flex
        flex_2 += 0.2 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_1 += 0.2 * curr_flex
        flex_2 += 1.0 * curr_flex
        flex_3 += 0.7 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_2 += 0.2 * curr_flex
        flex_3 += 1.0 * curr_flex
        return torch.tensor([flex_1, flex_2, flex_3]).type(torch.float)

    def _set_pinching_finger(self):
        def pertubation():
            return (torch.rand(1)*20 - 10) * pi / 180 
        def rand_flex():
            return (torch.rand(1) * 30 + 5) * pi / 180

        flex_1, flex_2, flex_3 = -0.2, -0.4, -0.34
        curr_flex = (torch.rand(1) * 30 + 60) * pi / 180 + pertubation()
        flex_1 += 1.0 * curr_flex
        flex_2 += 0.2 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_1 += 0.2 * curr_flex
        flex_2 += 1.0 * curr_flex
        flex_3 += 0.7 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_2 += 0.2 * curr_flex
        flex_3 += 1.0 * curr_flex
        return torch.tensor([flex_1, flex_2, flex_3]).type(torch.float)
    
    
    def _set_half_open_finger(self):
        def pertubation():
            return (torch.rand(1)*20 - 10) * pi / 180 
        def rand_flex():
            return (torch.rand(1) * 30 + 60) * pi / 180

        flex_1, flex_2, flex_3 = -0.2, -0.4, -0.34
        curr_flex = (torch.rand(1) * 30) * pi / 180 + pertubation()
        flex_1 += 1.0 * curr_flex
        flex_2 += 0.2 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_1 += 0.2 * curr_flex
        flex_2 += 1.0 * curr_flex
        flex_3 += 0.7 * curr_flex

        curr_flex = rand_flex() + pertubation()
        flex_2 += 0.2 * curr_flex
        flex_3 += 1.0 * curr_flex
        return torch.tensor([flex_1, flex_2, flex_3]).type(torch.float)

    def _set_open_finger(self):
        flex_1 = torch.rand(1) *0.25 - 0.1
        flex_2 = torch.rand(1) * 0.4 - 0.1
        flex_3 = torch.rand(1) * 0.34 - 0.1
        return torch.tensor([flex_1, flex_2, flex_3]).type(torch.float)
    
    def _set_straight_finger(self):
        flex_1 = torch.rand(1) *0.25 - 0.25
        flex_2 = torch.rand(1) * 0.4 - 0.4
        flex_3 = torch.rand(1) * 0.34 - 0.34
        return torch.tensor([flex_1, flex_2, flex_3]).type(torch.float)
    
    def _set_thumb(self):
        if torch.rand(1) < 0.5:
            flex = torch.rand(1) * 0.35 - 0.25
        else:
            flex = torch.rand(1) * 0.6 + 0.1

        flex_1 = flex
        flex_2 = 0.25 * flex
        flex_3 = torch.rand(1)*2 - 1.7

        abduct = torch.rand(1) - 0.5
        return torch.tensor([abduct, flex_1, flex_2, flex_3]).type(torch.float)
    
    def _set_rand_open_flex(self):
        mode = int(torch.rand(1)*3)
        if mode == 0:
            return self._set_straight_finger()
        elif mode == 1:
            return self._set_open_finger()
        elif mode == 2:
            return self._set_half_open_finger()
    
    def _set_rand_close_flex(self):
        mode = int(torch.rand(1)*2)
        if mode == 0:
            return self._set_pinching_finger()
        elif mode == 1:
            return self._set_closed_finger()

    def _set_rand_flex(self):
        mode = int(torch.rand(1)*5)
        if mode == 0:
            return self._set_straight_finger()
        elif mode == 1:
            return self._set_open_finger()
        elif mode == 2:
            return self._set_half_open_finger()
        elif mode == 3:
            return self._set_pinching_finger()
        elif mode == 4:
            return self._set_closed_finger()

    def _set_flex(self):
        # the random patterns of the fingers
        mode = int(torch.rand(1)*10)
        if mode == 0:
            #straight
            flex = [self._set_straight_finger() for idx in range(4)]
        elif mode == 1:
            #open
            flex = [self._set_open_finger() for idx in range(4)]
        elif mode == 2:
            #half open
            flex = [self._set_half_open_finger() for idx in range(4)]
        elif mode == 3:
            #pinching
            flex = [self._set_pinching_finger() for idx in range(4)]
        elif mode == 4:
            #closed
            flex = [self._set_closed_finger() for idx in range(4)]
        elif mode == 5:
            # index open
            flex = [self._set_rand_open_flex(), 
                    self._set_rand_close_flex(),
                    self._set_rand_close_flex(),
                    self._set_rand_close_flex()] 
        elif mode == 6:
            # pinky open
            flex = [self._set_rand_close_flex(), 
                    self._set_rand_close_flex(),
                    self._set_rand_close_flex(),
                    self._set_rand_open_flex()] 
        elif mode == 7:
            # index & middle open
            flex = [self._set_rand_open_flex(), 
                    self._set_rand_open_flex(),
                    self._set_rand_close_flex(),
                    self._set_rand_close_flex()] 
        elif mode == 8:
            # middle & ring & pinky open
            flex = [self._set_rand_close_flex(), 
                    self._set_rand_open_flex(),
                    self._set_rand_open_flex(),
                    self._set_rand_open_flex()] 
        elif mode == 8:
            # index & pinky open
            flex = [self._set_rand_open_flex(), 
                    self._set_rand_close_flex(),
                    self._set_rand_close_flex(),
                    self._set_rand_open_flex()] 
        elif mode == 9:
            # some really random state
             flex = [self._set_rand_flex(), 
                     self._set_rand_flex(),
                     self._set_rand_flex(),
                     self._set_rand_flex()]
        return flex
    
    def __getitem__(self, index: int):
        parameter = torch.zeros(self.num_parameter)
        parameter[0:6] = self._set_palm()
        abduct_angles = self._set_abduct()
        parameter[self.THUMB+self.ABDUCT : self.THUMB + self.FLEX_3+1] = self._set_thumb()

        parameter[self.INDEX + self.ABDUCT] = abduct_angles[0]
        parameter[self.MIDDLE + self.ABDUCT] = abduct_angles[1]
        parameter[self.RING + self.ABDUCT] = abduct_angles[2]
        parameter[self.PINKY + self.ABDUCT] = abduct_angles[3]

        finger_flex = self._set_flex()
        parameter[self.INDEX + self.FLEX_1 : self.INDEX + self.FLEX_3+1] = finger_flex[0]
        parameter[self.MIDDLE + self.FLEX_1 : self.MIDDLE + self.FLEX_3+1] = finger_flex[1]
        parameter[self.RING + self.FLEX_1 : self.RING + self.FLEX_3+1] = finger_flex[2]
        parameter[self.PINKY + self.FLEX_1 : self.PINKY + self.FLEX_3+1] = finger_flex[3]

        return parameter

    def __len__(self):
        return 400000



if __name__ == '__main__':
    batch_size = 64
    joint_angle_dataset = JointAngleDataset()
    data_loader = data.DataLoader(joint_angle_dataset, batch_size=64, num_workers=4, shuffle=True)

    import matplotlib.pyplot as plt
    from network.util_modules import HandSynthesizer
    import pickle

    with open('mesh/model/preprocessed_hand.pkl', 'rb') as f:
        mesh = pickle.load(f)
    hand_synthesizer = HandSynthesizer(mesh, 64, 16, 1.0, 0.01)
    hand_synthesizer = hand_synthesizer.cuda()


    data_it = iter(data_loader)
    for idx in range(1):
        parameter = next(data_it)
        parameter = parameter.cuda()
        dms, uv_hms, d_hms, xyz_pts = hand_synthesizer(parameter)
        
        figure = plt.figure()
        for idx in range(64):
            uv_hms = torch.nn.functional.interpolate(uv_hms, size=(64, 64))
            d_hms = torch.nn.functional.interpolate(d_hms, size=(64, 64))
            ax = figure.add_subplot(8, 8, idx+1)
            ax.imshow(dms[idx], vmin=-1.0, vmax=1.0,  cmap='gray')
            plt.axis('off')
        plt.show()
        if idx > 100:
            break
        