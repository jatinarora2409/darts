import argparse
import os, sys

sys.path.insert(0, '../cnn/')
import torch

from net.huffmancoding import huffman_encode_model
import test
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--model', type=str, default='../cnn/eval-EXP-20200415-092238/model_after_weight_sharing.ptmodel',
                    help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--output', default='saves/model_after_pruning_and_quantization.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = test.run_test(args.output,2)

# model = torch.load(args.model)
huffman_encode_model(model)
