import argparse

import torch

from net.huffmancoding import huffman_encode_model
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--model', type=str, default='../cnn/eval-EXP-20200415-092238/model_after_weight_sharing.ptmodel', help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = torch.load(args.model)
huffman_encode_model(model)