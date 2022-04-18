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

class BARTModel(pl.LightningModule):
	def __init__(self, argdict, datasets):
		super().__init__()

		self.argdict=argdict
		self.training_set, self.dev_set=datasets
		print(f"Training with {len(self.training_set)} exemples, and {len(self.training_set.index_unique_examples)} unique examples")
		print(f"Testing with {len(self.dev_set)} examples, and {len(self.dev_set.index_unique_examples)} unique examples")

		# self.device = torch.device(argdict['device'])

		# pretrained=['gpt', 'antoiloui/belgpt2']
		if argdict['language']=='en':
			gptPath='facebook/bart-large'
			self.tokenizer = BartTokenizer.from_pretrained(gptPath)
			model = BartForConditionalGeneration.from_pretrained(gptPath, cache_dir='/Tmp')
		elif argdict['language']=='fr':
			gptPath='moussaKam/mbarthez'
			self.tokenizer = AutoTokenizer.from_pretrained(gptPath)
			model = AutoModelForSeq2SeqLM.from_pretrained(gptPath, cache_dir='/Tmp')
		elif argdict['language']=='mu':
			gptPath = 'facebook/mbart-large-50'
			self.tokenizer = AutoTokenizer.from_pretrained(gptPath)
			model = AutoModelForSeq2SeqLM.from_pretrained(gptPath, cache_dir='/Tmp')
		else:
			raise ValueError("Unrecognized language")

		self.field_input='input_sentence'
		self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
		self.model=model#.to('cuda')#, config=config)
		self.model.config.max_length=argdict['max_seq_length']

		self.beam_search_k=10
		self.fuck_torch_lightning=[]
		self.fuck_torch_lightning_per_batch=[]

	def forward(self, tokenized_sentences, tokenized_decoder_sentences):

		input_ids=torch.Tensor(tokenized_sentences['input_ids']).long().to(self.device)
		attention_mask=torch.Tensor(tokenized_sentences['attention_mask']).to(self.device)
		decoder_input_ids=torch.Tensor(tokenized_decoder_sentences['input_ids']).long().to(self.device)
		decoder_attention_mask = torch.Tensor(tokenized_decoder_sentences['attention_mask']).to(self.device)
		#TODO ATTENTION MASK
		outputs=self.model(input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids, attention_mask=attention_mask)
		return outputs

	def training_step(self, batch, batch_idx):

		src = self.tokenizer(batch[self.field_input], padding=True, truncation=True, max_length=self.argdict['max_seq_length'])
		target = self.tokenizer(batch['full_labels'], padding=True, truncation=True)
		output = self.forward(src, target)
		loss = output['loss']
		self.log("Loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		return loss

	def validation_step(self, batch, batch_idx):
		src = self.tokenizer(batch[self.field_input], padding=True, truncation=True)
		target = self.tokenizer(batch['full_labels'], padding=True, truncation=True)
		output = self.forward(src, target)
		loss = output['loss']

		input_ids = self.tokenizer(batch[self.field_input], padding=True, truncation=True, return_tensors='pt', max_length=self.argdict['max_seq_length']).to(self.device)
		gend = self.model.generate(**input_ids, num_beams=10, num_return_sequences=1, max_length=50)
		gend = self.tokenizer.batch_decode(gend, skip_special_tokens=True)
		hypos=[self.score(sent) for sent in gend]
		inputs=batch[self.field_input]
		refs=[[rr.strip() for rr in fullLabels.split(',')] for fullLabels in batch['full_labels']]
		score = evaluate(inputs, refs, hypos, '<unk>', tokenizer='split_nopunc')
		f1 = np.average(score['all_exact_f_score@10'])
		prec = np.average(score['all_exact_precision@10'])
		rec = np.average(score['all_exact_recall@10'])

		self.fuck_torch_lightning_per_batch.append(f1)
		self.log("Loss_val", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		self.log("F1_val", f1, on_epoch=True, on_step=True, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		return loss

	def configure_optimizers(self):
		optimizer = AdamW(self.model.parameters(), lr=5e-5)
		return optimizer


	def score(self, generated_keyphrases, dev=False):
		#For dev, we test one input sentence at a time, we are not so difficult for train so we take only one sentence
		if not dev:
			dico = {}
			gg = generated_keyphrases.split(",")
			for i, word in enumerate(gg):
				word = word.strip()
				if word in dico.keys():
					dico[word].append(i + 1)
				else:
					dico[word] = [i + 1]
			# print(dico)
			for key, value in dico.items():
				try:
					score = sum([(1 / 11) + (1 / vv) for vv in value])
				except:
					# print(dico)
					fds
				dico[key] = score
			# print(dico)
			# print(sorted(dico, key=dico.__getitem__))
			# print(list(reversed(sorted(dico, key=dico.__getitem__))))
			return list(reversed(sorted(dico, key=dico.__getitem__)))[:10]
			# return list_output
		else:
			raise ValueError
		dico={}
		for gg in generated_keyphrases:
			gg=gg.split(",")
			for i, word in enumerate(gg):
				word=word.strip()
				if word in dico.keys():
					dico[word].append(i+1)
				else:
					dico[word]=[i+1]
		# print(dico)
		for key, value in dico.items():
			try:
				score=sum([(1/11)+(1/vv) for vv in value])
			except:
				# print(dico)
				fds
			dico[key]=score
		# print(dico)
		# print(sorted(dico, key=dico.__getitem__))
		# print(list(reversed(sorted(dico, key=dico.__getitem__))))
		return list(reversed(sorted(dico, key=dico.__getitem__)))[:10]

	def on_validation_epoch_end(self) -> None:
		# self.generate_from_dataset(2, 'train')
		# self.generate_from_dataset(2, 'dev')
		self.fuck_torch_lightning.append(np.mean(self.fuck_torch_lightning_per_batch))
		self.fuck_torch_lightning_per_batch=[]

	def train_model(self):
		# cb=MetricTracker()
		self.trainer=pl.Trainer(gpus=1, max_epochs=self.argdict['num_epochs'], precision=16, accumulate_grad_batches=64, enable_checkpointing=False)
		# trainer=pl.Trainer(max_epochs=self.argdict['num_epochs'])
		train_loader=DataLoader(
			dataset=self.training_set,
			batch_size=self.argdict['batch_size'],
			shuffle=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		dev_loader=DataLoader(
			dataset=self.dev_set,
			batch_size=self.argdict['batch_size'],
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		test_loader=DataLoader(
			dataset=self.test_set,
			batch_size=self.argdict['batch_size'],
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		self.trainer.fit(self, train_loader, dev_loader, )
		final=self.trainer.test(self, test_loader)
		# print("bitch")
		print(self.fuck_torch_lightning)
		print(final)
		# self.model.save_pretrained('Models/pretrained_bart')
		# for ep in range(self.argdict['num_epochs']):
		# 	loss, met10 = self.run_epoch('train')
		# 	loss_val,met10_val = self.run_epoch('dev')
		#
		# 	print(f'Epoch: {ep + 1:02}')
		# 	print(f'\tTrain Loss: {loss:.3f} | Metric @ 10 (p/r/f): {met10}')
		# 	print(f'\t Val. Loss: {loss_val:.3f} |  Metric @ 10 (p/r/f): {met10_val} ')
		# 	self.generate_from_dataset(split='train')
		# 	self.generate_from_dataset()

	def generate_from_dataset(self, n=2, split='dev'):
		"""Try to generate from the dev set"""
		self.model.eval()
		# self.model#.to('cuda')
		with torch.no_grad():
			print(f"Generation de {n} Notes Marginales from the {split} set")
			dataset=self.training_set if split=="train" else self.dev_set
			num_ex = len(dataset.index_unique_examples)
			prec_tot = 0
			rec_tot = 0
			f1_tot = 0
			inputs=[]
			refs=[]
			hypos=[]
			ll = self.argdict['max_seq_length']
			for j in range(num_ex):
				index = dataset.index_unique_examples[j]
				dat = dataset.data[index]
				refs.append(dat['full_labels'])
				inputs.append(dat[self.field_input])
				# src_text = " ".join(dat[self.field_input].split(' ')[:ll])
				# src_text = src_text
				input_ids = self.tokenizer.encode(dat[self.field_input], return_tensors='pt', truncation=True, max_length=self.argdict['max_seq_length']).to(self.device)
				# print(input_ids)
				# print(self.tokenizer.batch_decode((input_ids)))
				# fds
				# input_ids = torch.Tensor(src['input_ids']).long().to('cuda').unsqueeze(0)
				gend = self.model.generate(input_ids, num_beams=10, num_return_sequences=1,
									  max_length=50)
				# print(tokenizer.batch_decode(gend))
				gend = self.tokenizer.batch_decode(gend, skip_special_tokens=True)
				hypos.append(gend)
				if j==n:
					break

			# print(inputs, hypos, refs)

			for ii, hh, rr in zip(inputs, hypos, refs):
				print(f"Input : {ii} \n "
					  f"Note Marginale: {rr} \n"
					  f"Note Générée: {hh}")
				print("------------------")
		self.model.train()
