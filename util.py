"""Contain the data processor for Note Marginal for pytorch"""
from torch.utils.data import Dataset
import os, io
import numpy as np
import json
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from nltk.tokenize import TweetTokenizer, sent_tokenize
from torchtext.vocab import build_vocab_from_iterator



class OrderedCounter(Counter, OrderedDict):
	"""Counter that remembers the order elements are first encountered"""
	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

	def __reduce__(self):
		return self.__class__, (OrderedDict(self),)


def initialize_datasets(argdict):
	#Generate sentences: Prepare the dataset to generate sentences from the marginal notes
	train=pd.read_csv(f"data/{argdict['dataset']}/train.tsv", sep='\t', index_col=0)
	llPre=len(train)
	train=train.dropna()
	# train=train[:1000]
	# train=train[:10602]
	print(f"Dropped {len(train)-llPre} entries from train")
	dev=pd.read_csv(f"data/{argdict['dataset']}/dev.tsv", sep='\t', index_col=0)
	llPre=len(dev)
	dev=dev.dropna()
	if argdict['short_eval']:
		dev=dev[:100]
	print(f"Dropped {len(dev)-llPre} entries from dev")
	test = pd.read_csv(f"data/{argdict['dataset']}/test.tsv", sep='\t', index_col=0)
	test_inspec=pd.read_csv("data/Inspec/test.tsv", sep="\t", index_col=0)
	test_nus=pd.read_csv("data/NUS/test.tsv", sep="\t", index_col=0)
	test_semeval=pd.read_csv("data/SemEval/test.tsv", sep="\t", index_col=0).dropna()
	test_krapivin=pd.read_csv("data/krapivin/test.tsv", sep="\t", index_col=0).dropna()
	test_kp20k=pd.read_csv("data/kp20k/test.tsv", sep="\t", index_col=0).dropna()
	#Multilingual test sets
	test_wikinews=pd.read_csv("data/WikiNews/test.tsv", sep='\t', index_col=0).dropna()
	test_110ptbnkp=pd.read_csv("data/110ptbnkp/test.tsv", sep='\t', index_col=0).dropna()
	test_cacic57=pd.read_csv("data/cacic57/test.tsv", sep='\t', index_col=0).dropna()
	test_pak2018=pd.read_csv("data/pak2018/test.tsv", sep='\t', index_col=0).dropna()
	test_wicc78=pd.read_csv("data/wicc78/test.tsv", sep='\t', index_col=0).dropna()
	#Our own test sets
	test_papyruse = pd.read_csv("data/papyrus_e/test.tsv", sep="\t", index_col=0).dropna()
	test_papyrusf = pd.read_csv("data/papyrus_f/test.tsv", sep="\t", index_col=0).dropna()
	test_papyrusm = pd.read_csv("data/papyrus_m/test.tsv", sep="\t", index_col=0).dropna()
	test_papyrus = pd.read_csv("data/papyrus/test.tsv", sep="\t", index_col=0).dropna()
	llPre = len(test)
	test = test.dropna()
	if argdict['short_eval']:
		test=test[:100]
	print(f"Dropped {len(test) - llPre} entries from test")

	allsentences=list(train['sentences'])
	allsentences.extend(list(train['label']))
	# tokenizer=TweetTokenizer()
	# allsentences=[tokenizer.tokenize(sentence) for sentence in allsentences if sentence==sentence]
	# vocab = build_vocab_from_iterator(allsentences, min_freq=argdict['min_vocab_freq'], specials=["<unk>", "<pad>", "<bos>", "<eos>"], )
	# vocab.set_default_index(vocab["<unk>"])
	train=NoteMarg(train, argdict)
	dev=NoteMarg(dev, argdict, dev=True)
	test=NoteMarg(test, argdict, dev=True)
	test_inspec=NoteMarg(test_inspec, argdict, dev=True, no_index=True)
	test_nus=NoteMarg(test_nus, argdict, dev=True, no_index=True)
	test_semeval=NoteMarg(test_semeval, argdict, dev=True, no_index=True)
	test_krapivin=NoteMarg(test_krapivin, argdict, dev=True, no_index=True)
	test_papyruse=NoteMarg(test_papyruse, argdict, dev=True, no_index=True)
	test_papyrusf=NoteMarg(test_papyrusf, argdict, dev=True, no_index=True)
	test_papyrusm=NoteMarg(test_papyrusm, argdict, dev=True, no_index=True)
	test_papyrus=NoteMarg(test_papyrus, argdict, dev=True, no_index=True)
	test_kp20k=NoteMarg(test_kp20k, argdict, dev=True, no_index=True)
	test_wikinews=NoteMarg(test_wikinews, argdict, dev=True, no_index=True)
	test_110ptbnkp=NoteMarg(test_110ptbnkp, argdict, dev=True, no_index=True)
	test_cacic57=NoteMarg(test_cacic57, argdict, dev=True, no_index=True)
	test_pak2018=NoteMarg(test_pak2018, argdict, dev=True, no_index=True)
	test_wicc78=NoteMarg(test_wicc78, argdict, dev=True, no_index=True)
	return train, dev, {
						# "test_wicc78":test_wicc78,
						# "test_110ptbnkp":test_110ptbnkp,
						# "test_wikinews":test_wikinews,
						# "test_cacic57":test_cacic57,
						# "test_pak2018":test_pak2018,
						# "test_papyruse":test_papyruse,
						# "test_papyrusf":test_papyrusf,
						# "test_papyrusm":test_papyrusm,
						# "test_papyrus":test_papyrus,
						# "test_kp20k": test_kp20k,
						# "test_semeval":test_semeval,
						# "test_krapivin":test_krapivin,
						# "test_nus":test_nus,
						# "test_inspec":test_inspec,
						# "test":test
						}


class NoteMarg(Dataset):

	def __init__(self, data, argdict, dev=False, no_index=False):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.max_len = argdict['max_seq_length']
		# self.vocab = vocab
		# self.tokenizer=tokenizer
		# self.pad_idx = self.vocab['<pad>']
		self.max_len_label=0
		self.max_len_words=0
		self.num_sentences=0
		self.max_len_labels=0
		self.len_sentence=0
		self.index_unique_examples=[]
		# self.generate_sentence = generate_sentences
		index=0
		self.map_unique_to_id={}

		self.abstract_for_ex=[]
		self.label_for_ex=[]
		self.language_for_ex=[]

		if not dev:
			special_ex_df=pd.read_csv(f"data/papyrus_m/dev.tsv", sep='\t', index_col=0)
			for i, row in special_ex_df.iterrows():
				# print(row)
				if row['index'] == 24192:
					self.abstract_for_ex.append(row['sentences'])
					self.label_for_ex.append(row['label'])
		for i, row in data.iterrows():
			# Special example 26534
			if argdict['dataset'] not in ['kp20k'] and not dev and row['index'] == 25397:
				self.abstract_for_ex.append(row['sentences'])
				self.label_for_ex.append(row['label'])
			# print(row)
			if argdict['dataset'] not in ['kp20k'] and dev and not no_index and row['index'] == 24284:
				self.abstract_for_ex.append(row['sentences'])
				self.label_for_ex.append(row['label'])
				self.language_for_ex.append(row['language'])
			if dev and argdict['short_eval'] and index>10:
				break
			if row['sentences'] in ['.', '', ' '] or row['label'] in ['.', '', ' ']:
				continue
			# if self.len_sentence<max([len(sent) for sent in sentences_sep]):
			#     self.len_sentence=max([len(sent) for sent in sentences_sep])
			self.index_unique_examples.append(index)
			self.map_unique_to_id[index]=[]

			#Split by max_seq_length
			sents=sent_tokenize(row['sentences'])
			sents_trunc=[""]
			for ss in sents:
				if len(ss[-1].split(" "))+len(ss.split(" "))>argdict['max_seq_length']:
					sents_trunc.append(ss)
				else:
					sents_trunc[-1]+=ss+" "
			for sent in sents_trunc:
				if argdict['dataset'] not in ['kp20k'] and not no_index:
					ind=row['index']
				else:
					ind=0
				try:
					ll=row['language']
				except:
					ll='en'
				self.data[index] = {'full_labels': row['label'], 'input_sentence': sent, 'index':ind, 'language':ll}
				self.map_unique_to_id[self.index_unique_examples[-1]].append(index)
				index+=1

	def tokenize(self, batch):
		"""tokenize a batch"""
		results=[]
		for sent in batch:
			tokenized_text = self.tokenizer.tokenize(sent)
			results.append(self.vocab(tokenized_text))

		return results

	@property
	def vocab_size(self):
		return len(self.vocab)

	@property
	def eos_idx(self):
		return self.vocab['<eos>']

	@property
	def pad_idx(self):
		return self.vocab['<pad>']

	@property
	def bos_idx(self):
		return self.vocab['<bos>']

	@property
	def unk_idx(self):
		return self.vocab['<unk>']

	def get_i2w(self):
		return self.vocab.get_itos()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		# print(item)
		# print(self.data[item])
		return {
			'input_sentence':self.data[item]['input_sentence'],
			'full_labels':self.data[item]['full_labels'],
			'index':self.data[item]['index'],
			'language':self.data[item]['language']
		}


	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex
