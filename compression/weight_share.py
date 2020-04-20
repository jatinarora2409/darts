import argparse
import os,sys
sys.path.insert(0, '../cnn/') 
import torch

from net.models import LeNet
from net.quantization import apply_weight_sharing
import test
from model import NetworkCIFAR as Network

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('--model', type=str, default='../cnn/eval-EXP-20200415-092238/weights.pt', help='path to saved pruned model')
parser.add_argument('--img_cropped_height', type=int, default=32, help='img cropped height')
parser.add_argument('--img_cropped_width', type=int, default=32, help='img cropped width')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--output', default='../cnn/eval-EXP-20200415-092238/model_after_weight_sharing.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()


# Define the model
# utils.load(model, args.model_path)
# model = torch.load(args.model)
print('accuracy before weight sharing')
model = test.run_test(args.model)

# Weight sharing
apply_weight_sharing(model)
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)

print('accuacy after weight sharing')
test.run_test(args.output)

# Save the new model

