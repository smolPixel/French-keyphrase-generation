import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import math
import nltk
import nltk.translate.bleu_score as bleu
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, WarmUp, BartConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from datasets import load_metric
from eval import *
from tqdm import tqdm
# from transformers.modeling_bart import shift_tokens_right
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import timeit


class SeqToSeqModel(pl.LightningModule):
	def __init__(self, argdict, datasets):
		super().__init__()

		self.argdict=argdict
		self.training_set, self.dev_set, self.test_set=datasets
		print(f"Training with {len(self.training_set)} exemples, and {len(self.training_set.index_unique_examples)} unique examples")
		print(f"Testing with {len(self.dev_set)} examples, and {len(self.dev_set.index_unique_examples)} unique examples")


		input_dim=None
		output_dim=self.argdict['embed_size']
		"""Encoder"""
		self.embeddings=torch.nn.Embedding(input_dim, output_dim)
		self.rnn_encoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], self.argdict['num_layers'], batch_first=True, bidirectional=self.argdict['bidir_encoder'])
		"""Decoder"""
		self.embeddings=torch.nn.Embedding(input_dim, output_dim)
		self.rnn_decoder=torch.nn.GRU(self.argdict['embed_size'], self.argdict['hidden_size'], self.argdict['num_layers'], batch_first=True, bidirectional=False)
		"""Attention"""
		# self.attn=nn.Linear(self.argdict['hidden_size'])

	def forward(self, tokenized_sentences, tokenized_decoder_sentences):

		input_ids=torch.Tensor(tokenized_sentences['input_ids']).long().to(self.device)
		attention_mask=torch.Tensor(tokenized_sentences['attention_mask']).to(self.device)
		decoder_input_ids=torch.Tensor(tokenized_decoder_sentences['input_ids']).long().to(self.device)
		decoder_attention_mask = torch.Tensor(tokenized_decoder_sentences['attention_mask']).to(self.device)
		#TODO ATTENTION MASK
		outputs=self.model(input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids, attention_mask=attention_mask)
		return outputs