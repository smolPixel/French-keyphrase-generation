import argparse, random, torch
import numpy as np
from process_data import *
import subprocess
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from data.NoteMarg import NoteMarg


import math, time

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from process_data import *
from util import initialize_datasets
from AugmentStrat.Augmentator import Augmentator
from Model.BART import BARTModel

def run_external_process(process):
    output, error = process.communicate()
    if process.returncode != 0:
        raise SystemError
    return output, error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(argdict):
    datasets = initialize_NoteMarg(argdict, False)
    print(datasets)
    dsa
    model=BARTModel(argdict, datasets)
    model.train_model()
    model.generate_from_dataset(split='train')
    model.generate_from_dataset(split='dev')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Using Barthez for keyphrase generation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='NoteMarginales', help="dataset you want to run the process on. Includes NoteMarginales, NoteMarginalesFr")
    parser.add_argument('--computer', type=str, default='labo', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--algo', type=str, default='None', help='data augmentation algorithm to use, includes, EDA, ConstraintVAE, VAE, HierarchicalVAE')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='max length, 0 if no max length')


    parser.add_argument('--short_eval', action='store_true', help='Cut evaluation time for debugging purposes')

    args = parser.parse_args()
    argsdict = args.__dict__
    run(argsdict)