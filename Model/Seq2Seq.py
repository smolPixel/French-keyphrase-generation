import random
from typing import Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import timeit
from nltk.tokenize import TweetTokenizer, sent_tokenize
from torchtext.vocab import build_vocab_from_iterator
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, WarmUp, BartConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from Model.Models import SeqToSeq
from torch.nn.utils.rnn import pad_sequence

class SeqToSeqModel(pl.LightningModule):
	def __init__(self, argdict, datasets):
		super().__init__()
		self.argdict=argdict
		self.training_set, self.dev_set, self.test_set=datasets
		print(f"Training with {len(self.training_set)} exemples, and {len(self.training_set.index_unique_examples)} unique examples")
		print(f"Testing with {len(self.dev_set)} examples, and {len(self.dev_set.index_unique_examples)} unique examples")

		self.field_input = 'input_sentence'
		#Tokenizer
		self.tokenizer = TweetTokenizer()
		allsentences=list(self.training_set.df['sentences'])
		allsentences = [self.tokenizer.tokenize(sentence) for sentence in allsentences if sentence == sentence]
		specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
		self.vocab = build_vocab_from_iterator(allsentences, specials=specials)
		self.vocab.set_default_index(self.vocab["<unk>"])
		self.stoi=self.vocab.get_stoi()
		input_dim=len(self.vocab)
		self.argdict['input_size']=input_dim
		output_dim=self.argdict['embed_size']
		self.model=SeqToSeq(argdict)

	def configure_optimizers(self):
		optimizer = AdamW(self.model.parameters(), lr=5e-5)
		return optimizer

	def training_step(self, batch, batch_idx):
		src = self.tokenizer(batch[self.field_input], padding=True, truncation=True, max_length=self.argdict['max_seq_length'])
		target = self.tokenizer(batch['full_labels'], padding=True, truncation=True)
		output = self.forward(src, target)
		loss = output['loss']
		self.log("Loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		return loss

	def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
		text_batch = batch[self.field_input]
		tokenized=[torch.Tensor([int(self.vocab[token]) for token in self.tokenizer.tokenize(sent)]) for sent in text_batch]
		input_ids=pad_sequence(tokenized, batch_first=True, padding_value=self.vocab.get_default_index())
		batch['input_ids'] = input_ids

	def validation_step(self, batch, batch_idx):
		src = self.tokenizer(batch[self.field_input], padding=True, truncation=True)
		target = self.tokenizer(batch['full_labels'], padding=True, truncation=True)
		output = self.forward(src, target)
		loss = output['loss']

		input_ids = self.tokenizer.tokenize(batch[self.field_input], padding=True, truncation=True, return_tensors='pt', max_length=self.argdict['max_seq_length']).to(self.device)
		gend = self.model.generate(**input_ids, num_beams=10, num_return_sequences=1, max_length=50)
		gend = self.tokenizer.batch_decode(gend, skip_special_tokens=True)
		hypos=[self.score(sent) for sent in gend]
		inputs=batch[self.field_input]
		refs=[[rr.strip() for rr in fullLabels.split(',')] for fullLabels in batch['full_labels']]
		score = evaluate(inputs, refs, hypos, '<unk>', tokenizer='split_nopunc')
		# print(score)
		f110 = np.average(score['present_exact_f_score@10'])
		f15 = np.average(score['present_exact_f_score@5'])
		r10 = np.average(score['absent_exact_recall@10'])
		# prec = np.average(score['all_exact_precision@10'])
		# rec = np.average(score['all_exact_recall@10'])
		# score5=evaluate(inputs, refs, [sents[:5] for sents in hypos], '<unk>', tokenizer='split_nopunc')
		# f15 = np.average(score5['present_exact_f_score@5'])

		self.logger_per_batch.append((f15, f110, r10))
		self.log("Loss_val", loss, on_epoch=True, on_step=False, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		self.log("F1_val_10", f110, on_epoch=True, on_step=False, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		self.log("F1_val_5", f15, on_epoch=True, on_step=False, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		self.log("r_val_5", f15, on_epoch=True, on_step=False, prog_bar=True, logger=False, batch_size=self.argdict['batch_size'])
		return loss
	def forward(self, tokenized_sentences, tokenized_decoder_sentences):

		input_ids=torch.Tensor(tokenized_sentences['input_ids']).long().to(self.device)
		attention_mask=torch.Tensor(tokenized_sentences['attention_mask']).to(self.device)
		decoder_input_ids=torch.Tensor(tokenized_decoder_sentences['input_ids']).long().to(self.device)
		decoder_attention_mask = torch.Tensor(tokenized_decoder_sentences['attention_mask']).to(self.device)
		#TODO ATTENTION MASK
		outputs=self.model(input_ids, decoder_attention_mask=decoder_attention_mask, labels=decoder_input_ids, attention_mask=attention_mask)
		return outputs

	def train_model(self):
		# cb=MetricTracker()
		early_stopping_callback = EarlyStopping(monitor='F1_val_10', patience=1, mode='max')
		self.trainer = pl.Trainer(gpus=1, max_epochs=self.argdict['num_epochs'], precision=16,
								  accumulate_grad_batches=self.argdict['accum_batch_size'], enable_checkpointing=False)
		train_loader = DataLoader(
			dataset=self.training_set,
			batch_size=self.argdict['batch_size'],
			shuffle=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		dev_loader = DataLoader(
			dataset=self.dev_set,
			batch_size=self.argdict['batch_size'],
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)


		path_save = f'/data/rali6/Tmp/piedboef/Models/FKPG/{self.argdict["dataset"]}_SeqToSeq_{self.argdict["num_epochs"]}Epochs_random_seed_{self.argdict["random_seed"]}.pt'

		try:
			self.model.load_state_dict(torch.load(path_save))
			print("loaded model")
		except:
			tic = timeit.default_timer()
			self.trainer.fit(self, train_loader, dev_loader)
			print("saving model")
			# torch.save(self.model.state_dict(), path_save)
			print(self.loggerg)
			toc = timeit.default_timer()
			print(f"Training processed took {toc - tic} seconds")
			return 0
		# self.generate_special_ex()
		for name, tt in self.test_set.items():
			if name in ['test_semeval', 'test_inspec', 'test_nus', 'test_kp20k', 'test_papyruse', 'test_krapivin',
						'test_wikinews']:
				self.testing_standard_dataset = True
			else:
				self.testing_standard_dataset = False
			test_loader = DataLoader(
				dataset=tt,
				batch_size=self.argdict['batch_size'],
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			print(f"Running test for {name}")
			# final=self.trainer.test(self, test_loader)
			self.compare_correct_kp(test_loader)