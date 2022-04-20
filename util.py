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
    print(f"Dropped {len(train)-llPre} entries from train")
    dev=pd.read_csv(f"data/{argdict['dataset']}/dev.tsv", sep='\t', index_col=0)
    llPre=len(dev)
    dev=dev.dropna()
    # dev=dev[:100]
    print(f"Dropped {len(dev)-llPre} entries from dev")
    test = pd.read_csv(f"data/{argdict['dataset']}/test.tsv", sep='\t', index_col=0)
    llPre = len(test)
    test = test.dropna()
    # test=test[:100]
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
    return train, dev, test

class NoteMarg(Dataset):

    def __init__(self, data, argdict, dev=False):
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




        for i, row in data.iterrows():
            # Special example 26534
            if not dev and row['index'] == 22634:
                self.abstract_for_ex = row['sentences']
                self.label_for_ex = row['label']
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
                self.data[index] = {'full_labels': row['label'], 'input_sentence': sent}
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
        }


    def iterexamples(self):
        for i, ex in self.data.items():
            yield i, ex
