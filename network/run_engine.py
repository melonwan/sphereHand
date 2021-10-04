from __future__ import absolute_import, division, print_function
import argparse
import torch
from network.engine import Engine

default_dataset_path = 'E:\\data\\nyu\\npy-64'
default_model_path = 'E:\\exp\\trained_model\\'

parser = argparse.ArgumentParser()
parser.add_argument('--synthesize', default=True, action='store_false')
parser.add_argument('--mv_projection', default=True, action='store_false')
parser.add_argument('--mv_consistency', default=True, action='store_false')
parser.add_argument('--temporal', default=False, action='store_true')
parser.add_argument('--collision', default=True, action='store_false')
parser.add_argument('--bone_length', default=True, action='store_false')
parser.add_argument('--prior', default=True, action='store_false')
parser.add_argument('--mode', default='Test', type=str)
parser.add_argument('--model_dir', default=default_model_path, type=str)
parser.add_argument('--initial_model', type=str)
parser.add_argument('--restore_from_model', type=str)
parser.add_argument('--restore_from_epoch', default='-1', type=int)
parser.add_argument('--num_stacks', default=1, type=int)
parser.add_argument('--epoch', default=75, type=int)
parser.add_argument('--dataset_dir', 
                    default=default_dataset_path, 
                    type=str)
parser.add_argument('--depth_resample', default=0, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--tag', default='', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    engine = Engine(args)
    if args.mode == 'Train':
        engine.train()
    else:
        engine.eval()
