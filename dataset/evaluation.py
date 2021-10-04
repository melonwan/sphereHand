from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy.linalg as alg
import numpy as np
import pickle
import os

class Evaluation(object):
    def __init__(self):
        # self.synt_key_points = [33,32, 27,26, 21,20, 15,14, 39,40,38, 0,1,2]
        # self.real_key_points = [0,3, 6,9, 12,15, 18,21, 24,25,27, 30,31,32]

        self.synt_key_points = [33,32, 27,26, 21,20, 15,14, 39,40,38,2]
        self.real_key_points = [0,3, 6,9, 12,15, 18,21, 24,25,27,32]

        # self.synt_key_points = [33, 27, 21, 15, 39]
        # self.real_key_points = [0, 6, 12, 18, 24]

    def estimate_from_file(self, path: str):
        with open(path, 'rb') as f:
            results = pickle.load(f)
            gt_joints = results['gt']
            est_joints = results['est']

            if gt_joints.ndim == 4:
                num_batch = gt_joints.shape[0]
                num_views = gt_joints.shape[1]
                num_joints = gt_joints.shape[2]
                gt_joints = gt_joints.reshape(num_batch*num_views, num_joints, 3)
            if est_joints.ndim == 4:
                num_batch = est_joints.shape[0]
                num_views = est_joints.shape[1]
                num_joints = est_joints.shape[2]
                est_joints = est_joints.reshape(num_batch*num_views, num_joints, 3)
            gt_joints = gt_joints[:, self.real_key_points]
            est_joints = est_joints[:, self.synt_key_points]

        dir_name = os.path.dirname(path)
        error = alg.norm(gt_joints - est_joints, axis=-1).mean(axis=0)
        with open(os.path.join(dir_name, 'per_joint_mean_error.txt'), 'w') as f:
            for idx, e in enumerate(error):
                f.write('{}: {}\n'.format(idx, e))

        max_error = []
        mean_error = []
        for gt_j, est_j in zip(gt_joints, est_joints):
            max_error.append(Evaluation.maxJntError(gt_j, est_j))
            mean_error.append(Evaluation.meanJntError(gt_j, est_j))
        
        plot_path = os.path.join(dir_name, 'max_error')
        print('average error: ', np.asarray(mean_error).mean())
        with open(os.path.join(dir_name, 'mean_error.txt'), 'w') as f:
            f.write('average error: {}\n'.format(np.asarray(mean_error).mean()))
        Evaluation.plotError(max_error, plot_path)

    @classmethod
    def maxJntError(cls_obj, skel1: np.ndarray, skel2: np.ndarray):
        diff = skel1.reshape(-1,3) - skel2.reshape(-1,3)
        diff = alg.norm(diff, axis=1)
        return diff.max() 

    @classmethod
    def meanJntError(cls_obj, skel1: np.ndarray, skel2: np.ndarray):
        diff = skel1.reshape(-1,3) - skel2.reshape(-1,3)
        diff = alg.norm(diff, axis=1)
        return diff.mean() 
    
    @classmethod
    def averageMaxJntError(cls_obj, score_list):
        score_list = sorted(score_list)
        thresh_list = [thresh*5.0+0.5 for thresh in range(0, 17)]
        precent_list = [1]*len(thresh_list)
        for i in range(0, len(thresh_list)):
            th_idx = 0
            for j in range(0, len(score_list)):
                if(score_list[j]<thresh_list[i]):
                    th_idx += 1
            precent_list[i] = float(th_idx) / len(score_list)
        return (thresh_list, precent_list)

    @classmethod
    def plotError(cls_obj, score_list, res_path):
        score_list = sorted(score_list)

        thresh_list = [thresh*5.0+0.5 for thresh in range(0, 17)]
        precent_list = [1]*len(thresh_list)
        for i in range(0, len(thresh_list)):
            th_idx = 0
            for j in range(0, len(score_list)):
                if(score_list[j]<thresh_list[i]):
                    th_idx += 1
            precent_list[i] = float(th_idx) / len(score_list)

        plt.clf()
        plt.plot(thresh_list, precent_list)
        plt.grid(True)
        plt.xlabel('max error thresh(mm)')
        plt.ylabel('percentage')
        plt.title('max joint error')
        plt.savefig(res_path+'.png')
        f = open(res_path+'.txt', 'w')
        for thresh, p in zip(thresh_list, precent_list):
            f.write('%f %f\n'%(thresh, p*100.))
        f.write('{}\n'.format(precent_list))
        f.close()

if __name__ == '__main__':
    res_dir = 'D:\\exp\\trained_model\\'
    res_tag = 's2'
    path = os.path.join(res_dir, res_tag, 'result.pkl')

    evaluator = Evaluation()
    evaluator.estimate_from_file(path)