import argparse
import os
import sys

sys.path.insert(0, '../cnn/')
sys.path.insert(0, 'net/')

import torch
import torch.nn.utils.prune as prune
from tensorflow.python.ops import nn
from torch import nn

from cnn import test
from cnn.model import Cell, ReLUConvBN

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--model', type=str, default='../cnn/eval-EXP-20200415-092238/weights.pt',
                    help='path to saved pruned model')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--output', default='saves/model_after_pruning.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()


def prune_darts(model, pruning_percentage):
    for modules in model.children():
        if not isinstance(modules, nn.AdaptiveAvgPool3d):
            for module in modules:
                if not isinstance(module, Cell):
                    # print(list(module.named_parameters()))
                    prune.random_unstructured(module, name="weight", amount=pruning_percentage)
                    # print(list(module.named_parameters()))
                    # print("Not Cell")
                else:
                    # print(module)
                    for cell_module in module.children():
                        # print("Cell")
                        if isinstance(cell_module, ReLUConvBN):
                            # print("ReLUConvBN")
                            for ReLU_List in cell_module.children():
                                for ReLU_module in ReLU_List:
                                    if isinstance(ReLU_module, nn.Conv2d):
                                        # if ReLU_module.kernel_size[0] == 1:
                                        #     continue
                                        prune.l1_unstructured(ReLU_module, name="weight", amount=pruning_percentage)
                        elif isinstance(cell_module, nn.ModuleList):
                            # print("nn.ModuleList")
                            for innerModules in cell_module.children():
                                for seqItem in innerModules.children():
                                    for layer in seqItem:
                                        if isinstance(layer, nn.Conv2d):
                                            # if layer.kernel_size[0] == 1:
                                            #     continue
                                            prune.l1_unstructured(layer, name="weight", amount=pruning_percentage)


# Control Seed
# torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

model = test.run_test(args.model, 1)

# NOTE : `weight_decay` term denotes L2 regularization loss term
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
# initial_optimizer_state_dict = optimizer.state_dict()

# model.prune_by_std(args.sensitivity)

# Pruning
# print("--- Before pruning ---")
var1 = 0.7
# prune_darts(model, var1)

for i2 in [0.8, 0.9]:
    model = test.run_test(args.model, 1)
    prune_darts(model, i2)
    print("--- After pruning --- ")
    print(i2)
    torch.save(model, args.output)
    test.run_test(args.output, 2)

# Retrain

# torch.save(model, args.output)
# test.run_test(args.output, 2)

