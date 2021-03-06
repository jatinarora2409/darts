import sys

sys.path.insert(0, '../cnn/')
sys.path.insert(0, 'net/')
import numpy as np
import torch
import genotypes
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from denoise_dataset import DENOISE_DATASET

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/mnt/d/SIDD_Medium_Srgb/Data',
                    help='location of the data corpus')
parser.add_argument('--train_data', type=str, default='../data/train_data/', help='location of the train_data corpus')
parser.add_argument('--label_data', type=str, default='../data/label_data/', help='location of the test_data corpus')
parser.add_argument('--img_cropped_height', type=int, default=32, help='img cropped height')
parser.add_argument('--img_cropped_width', type=int, default=32, help='img cropped width')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='./eval-EXP-20200415-092238/weights.pt',
                    help='path of pretrained model')
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


def run_test(model_data, var=1):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)

    genotype = genotypes.DENOISE_GENO

    model = Network(args.init_channels, 10, args.layers, args.auxiliary, genotype,
                    output_height=args.img_cropped_height, output_width=args.img_cropped_width)

    if var == 2:
        model = torch.load(model_data)
        model.cuda()
    else:
        model = model.cuda()
        utils.load(model, model_data)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.MSELoss()
    criterion = criterion.cuda()

    _, test_transform = utils._data_trainsforms_denosining_dataset(args)
    # utils._data_transforms_cifar10(args)

    test_data = DENOISE_DATASET(root=args.data, train_folder=args.train_data, label_folder=args.label_data, train=False,
                                transform=test_transform, target_transform=test_transform)
    # dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    # print(len(test_queue))

    model.drop_path_prob = args.drop_path_prob
    try:
        if var == 1:
            print('no test')
#        else:
#            test_acc, _ = infer(test_queue, model, criterion)
 #           logging.info('test_acc %f', test_acc)
    # except Exception as e:
    #     print('Failed')
    #     print(e)
    finally:
        return model


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    model.eval()
    for step, (input, target) in enumerate(test_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)

                # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                # n = input.size(0)
                # objs.update(loss.data[0], n)
                # top1.update(prec1.data[0], n)
                # top5.update(prec5.data[0], n)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        n = input.size(0)
        objs.update(loss.data, n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e', step, objs.avg)
        if step == 1350:
            print('Done with test accuracy')
            return objs.avg
    return objs.avg


if __name__ == '__main__':
    run_test()
