"""
Codebase for the paper: 
"On the Transferability of Visual Features in Generalized Zero-Shot Learning".

TLDR; Large Scale Benchmark for Generalized Zero-Shot Learning (GZSL).
Three different GZSL families, eight methods, three datasets, eighteen different feature backbones, fifty four visual features.
"""

from builtins import breakpoint
import datetime
import argparse
import wrapper as super_glue
from utils.general_config import *

parser = argparse.ArgumentParser(description='Large Scale Benchmark for Generalized Zero-Shot Learning')

parser.add_argument('--dataset', metavar='DATASET', default='CUB',
                        choices=['CUB', 'SUN', 'AWA2'],
                        help='dataset: CUB, SUN, AWA2')
parser.add_argument('--feature_backbone', default='resnet101',
                        choices=all_possible_archs_and_ft_features,
                        help='select feature backbone from' + ', '.join(all_possible_archs_and_ft_features))
parser.add_argument('--methods', default='CADA',
                        choices=all_methods,
                        help='default method: CADA -- select a method from' + ', '.join(all_methods))
parser.add_argument('--finetuned_features', action='store_true', help='If the RN101 features to use are finetuned')                        

parser.add_argument('--data_path', type = str, default = 'data',
                        help='folder where data is stored')                                         
parser.add_argument('--workers', default=12, type=int,
                    help='number of cpus for data loading')                        

parser.add_argument('--dropout', default=0.0, type=float, metavar='DO', help='dropout rate')
parser.add_argument('--optimizer', type = str, default = 'adam',
                    help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--initial_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=210, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate \
                    reaches to zero')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--doParallel', dest='doParallel', action='store_true',
                    help='use DataParallel')                      

# parser.add_argument('--checkpoint_epochs', default=50, type=int,
#                     metavar='EPOCHS', help='checkpoint frequency (by epoch)')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
                    
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--add_name', type=str, default='Zero_Shot_T1')
parser.add_argument('--exp_dir', type = str, default = '',
                        help='folder where the experiment is stored')

parser.add_argument('--load_from_epoch', default=1, type=int, help='set number of epoch to load saved checkpoint')

parser.add_argument('--seed', dest = 'seed', default=0, type=int, help='define seed for random distribution of dataset')

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')

if __name__ == '__main__':

    # parse arguments
    args = parser.parse_args()

    # create wrapper and prepare datasets
    wrapper = super_glue.Wrapper(args)

    # set model hyperparameters depending on method and dataset
    wrapper.set_model_hyperparameters()
    wrapper.set_model_optimizer()

    # algorithm train and validation calls
    wrapper.train()
    wrapper.train_classifier()



    

    