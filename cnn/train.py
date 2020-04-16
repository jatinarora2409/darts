import os
import sys
import time
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
parser.add_argument('--train_data', type=str, default='../data/train_data/', help='location of the train_data corpus')
parser.add_argument('--label_data', type=str, default='../data/label_data/', help='location of the test_data corpus')
parser.add_argument('--img_cropped_height', type=int, default=32, help='img cropped height')
parser.add_argument('--img_cropped_width', type=int, default=32, help='img cropped width')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DENOISE_GENO', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.MSELoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_trainsforms_denosining_dataset(args)
  train_data = DENOISE_DATASET(root=args.data,train_folder=args.train_data,label_folder=args.label_data,train=True, transform=train_transform,target_transform=train_transform )
 #valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_obj %f', train_obj)

    valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_obj %f', valid_obj)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    #prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)

    # if step % args.report_freq == 0:
    #   logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    #prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data, n)
    # if step % args.report_freq == 0:
    #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return objs.avg


if __name__ == '__main__':
  main() 

