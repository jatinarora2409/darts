import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from denoise_dataset import DENOISE_DATASET


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/mnt/d/SIDD_Medium_Srgb/Data', help='location of the data corpus')
parser.add_argument('--img_cropped_height', type=int, default=32, help='img cropped height')
parser.add_argument('--img_cropped_width', type=int, default=32, help='img cropped width')
parser.add_argument('--result_data',default='../result_images/', help='location of the data corpus')
parser.add_argument('--train_data', type=str, default='../data/train_data/', help='location of the train_data corpus')
parser.add_argument('--label_data', type=str, default='../data/label_data/', help='location of the test_data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='eval-EXP-20200415-092238/weights.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DENOISE_GENO', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype,output_height=args.img_cropped_height,output_width=args.img_cropped_width)
  model = model.cuda()
  utils.load(model, args.model_path)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.MSELoss()
  criterion = criterion.cuda()

  test_transform, test_valid_transform = utils._data_trainsforms_denosining_dataset(args)
  test_data = DENOISE_DATASET(root=args.data,train_folder=args.train_data,label_folder=args.label_data,train=True, transform=test_transform,target_transform=test_transform )

  # _, test_transform = utils._data_transforms_cifar10(args)
  # test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = args.drop_path_prob
  test_acc = infer(test_queue, model, criterion)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  model.eval()
  transformer =  utils._data_back_trainsform_dataset()
  batch_counter = 0
  for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    logits, _ = model(input)
    logit_shape = logits.shape

    for i in range(0,logit_shape[0]):
        result_img = transformer(logits[i].cpu())
        print(result_img.size)
        result_img.save(args.result_data+str(batch_counter)+"_"+str(i)+".png", "PNG")

    loss = criterion(logits, target)
    n = input.size(0)
    objs.update(loss.data, n)
    print(objs.avg)
    batch_counter  = batch_counter + 1;

  return objs.avg


if __name__ == '__main__':
  main()