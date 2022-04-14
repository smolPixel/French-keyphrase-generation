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
from data.NoteMarg import initialize_NoteMarg
from AugmentStrat.Augmentator import Augmentator

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
    # generate_sentence = argdict['classifier']=='gpt'
    createFolders(argdict)
    check_data_selected(argdict)
    augmentator=Augmentator(argdict)
    #TODO HERE YOU WANT IT TO WORK EVEN IF SPLIT IS 0
    dd=augmentator.augment()
    #Build the dataset with augmented data
    datasets = initialize_NoteMarg(argdict, False)
    # print(len(dd['train']))
    if argdict['classifier']=='LogReg':
        from Model.LogReg import LogReg
        model=LogReg(argdict, datasets)
    elif argdict['classifier'].lower()=='tfidf':
        from Model.TFIDF import TFIDFModel
        model=TFIDFModel(argdict, datasets)
    elif argdict['classifier'].lower()=='yake':
        from Model.YAKE import YAKEModel
        model=YAKEModel(argdict, datasets)
    if argdict['classifier']=='lstm':
        from Model.Seq_to_seq import Encoder, Attention, Decoder, Seq2Seq
        model = Seq2Seq(argdict, datasets)
    elif argdict['classifier']=='WordsLSTM':
        from Model.WordSeq_to_Seq import WordSeq2Seq
        model= WordSeq2Seq(argdict, datasets)
    elif argdict['classifier'].lower()=='bart':
        from Model.BART import BARTModel
        model=BARTModel(argdict, datasets)
    #We don't talk about this
    # elif argdict['classifier'] == 'bert':
    #     from Model.BertSeq_to_seq import BertSeq2Seq
    #     model = BertSeq2Seq(argdict, dd.datasets)
    elif argdict['classifier'] == 'gpt':
        from Model.GPT2 import GPT2Model
        model=GPT2Model(argdict, datasets)
    elif argdict['classifier']=='dummy':
        from Model.dummy import dummy
        model=dummy(argdict, dd.datasets)
    elif argdict['classifier']=='opennmt':
        from Model.OpenNMT import OpenNMT
        model=OpenNMT(argdict, datasets)
    elif argdict['classifier'].lower()=='gan':
        from Model.GAN import GAN
        model=GAN(argdict, datasets)
    if argdict['embedding'] in ['glove']:
        model.init_embeds()
    model.train_model()
    model.generate_from_dataset(split='train')
    model.generate_from_dataset(split='dev')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE for data augmentation')
    #General arguments on training
    parser.add_argument('--dataset', type=str, default='NoteMarginales', help="dataset you want to run the process on. Includes NoteMarginales, NoteMarginalesFr")
    parser.add_argument('--language', type=str, default='en', help="Language of the dataset")
    parser.add_argument('--classifier', type=str, default='lstm', help="classifier you want to use. Includes lstm, gpt, WordsLSTM, or dummy")
    parser.add_argument('--computer', type=str, default='labo', help="Whether you run at home or at iro. Automatically changes the base path")
    parser.add_argument('--split', type=float, default=0, help='percent of the dataset added')
    parser.add_argument('--retrain', action='store_true', help='whether to retrain the VAE or not')
    parser.add_argument('--rerun', action='store_true', help='whether to rerun knowing that is has already been ran')
    parser.add_argument('--dataset_size', type=int, default=10, help='number of example in the original dataset. If 0, use the entire dataset')
    parser.add_argument('--algo', type=str, default='None', help='data augmentation algorithm to use, includes, EDA, ConstraintVAE, VAE, HierarchicalVAE')
    parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--strategy', type=str, default='one2many', help='one2one or one2many')

    # Constraint VAE parameters
    parser.add_argument('--nb_epoch_algo', type=int, default=30, help='Number of epoch of the algo')
    parser.add_argument('--batch_size_algo', type=int, default=8, help='batch size of the data augmentation algo')
    parser.add_argument('--latent_size', type=int, default=5, help='Latent Size')
    parser.add_argument('--hidden_size_algo', type=int, default=1024, help='Hidden Size Algo')
    parser.add_argument('--dropout_algo', type=float, default=0.3, help='dropout of the classifier')
    parser.add_argument('--word_dropout', type=float, default=0.3, help='dropout of the classifier')
    parser.add_argument('--x0', default=2500, type=int, help='x0')
    parser.add_argument('--k', default=0.0025, type=float, help='k')
    parser.add_argument('--min_vocab_freq', default=1, type=int, help='min freq of vocab to be included')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--num_epochs_secondary', type=int, default=10, help='Number of epochs for secondary mechanisme (VAE in VAE_GPT)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--max_seq_length', type=int, default=64, help='max length, 0 if no max length')
    parser.add_argument('--embedding_size', type=int, default=300, help='size of the embedding')
    parser.add_argument('--embedding', type=str, default='none', help='either glove or none')
    parser.add_argument('--hidden_size', type=int, default=64, help='size of the hidden size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')

    parser.add_argument('--short_eval', action='store_true', help='Cut evaluation time for debugging purposes')



    # parser.add_argument('--path', type=str, default="/media/frederic/DAIro", help="Empirical Path to the folder")
    # parser.add_argument('--computer', type=str, default="home", help="home or labo")

    args = parser.parse_args()

    argsdict = args.__dict__


    if argsdict['dataset'] in ['NoteMarginalesFr']:
        argsdict['language']='fr'

    if argsdict['classifier'] in ['bert', 'gpt', 'bart']:
        print("transformers model detected, setting tokenizer to bert")
        argsdict['embedding']='bert'

    if argsdict['classifier'] in ['bart']:
        print("Bart Model detected, setting preprocessing to bart")
        argsdict['preprocessing']='bart'
    elif argsdict['classifier'] in ['gpt']:
        print("GPT Model detected, setting preprocessing to gpt")
        argsdict['preprocessing']='gpt'
    else:
        argsdict['preprocessing']=None


    if argsdict['computer']=='home':
        argsdict['path']='/media/frederic/DAIro'
    elif argsdict['computer']=='labo':
        argsdict['path']='/u/piedboef/Documents/DAIro'

    run(argsdict)