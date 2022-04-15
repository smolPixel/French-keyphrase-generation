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
    train=pd.read_csv("data/French/train.tsv", sep='\t', index_col=0)
    train=train.dropna()
    dev=pd.read_csv("data/French/dev.tsv", sep='\t', index_col=0)
    allsentences=list(train['sentences'])
    allsentences.extend(list(train['label']))
    # tokenizer=TweetTokenizer()
    # allsentences=[tokenizer.tokenize(sentence) for sentence in allsentences if sentence==sentence]
    # vocab = build_vocab_from_iterator(allsentences, min_freq=argdict['min_vocab_freq'], specials=["<unk>", "<pad>", "<bos>", "<eos>"], )
    # vocab.set_default_index(vocab["<unk>"])
    train=NoteMarg(train, argdict)
    dev=NoteMarg(dev, argdict, dev=True)
    return train, dev

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
        self.generate_sentence = generate_sentences
        index=0
        self.map_unique_to_id={}
        for i, row in data.iterrows():
            if dev and argdict['short_eval'] and index>10:
                break
            if row['sentences'] in ['.', '', ' '] or row['label'] in ['.', '', ' ']:
                continue
            # if self.len_sentence<max([len(sent) for sent in sentences_sep]):
            #     self.len_sentence=max([len(sent) for sent in sentences_sep])
            try:
                tokenized_text = self.tokenizer.tokenize("<bos> " + row['sentences'] + " <eos>")
            except:
                print("bish")
                print(row)
                fds
            input = np.array(vocab(tokenized_text))
            self.index_unique_examples.append(index)
            self.map_unique_to_id[index]=[]

            if argdict['one_kp']:
                labs=[row['label']]
            else:
                labs=row['label'].split(', ')
            # gptvae_sentence = ""

            # print(labs)
            # print(row['sentences']
            sents=sent_tokenize(row['sentences'])
            sents_trunc=[""]
            for ss in sents:
                if len(ss[-1].split(" "))+len(ss.split(" "))>argdict['max_seq_length']:
                    sents_trunc.append(ss)
                else:
                    sents_trunc[-1]+=ss+" "
            for sent in sents_trunc:
                sentence_max_len = sent
                if generate_sentences:
                    sent_gpt = "<bos> " + row['label'].lower() + " <sep> " + row['sentences'].lower() + " <eos>"
                    # gptvae_sentence="<bos> "+row['sentences'].lower() +" <sep> "+row['label'].lower()+" <eos>"
                    All_Labels_Tokenized = np.array(
                        vocab(self.tokenizer.tokenize("<bos> " + row['label'] + " <eos>")))
                    # print(All_Labels_Tokenized)
                    if len(All_Labels_Tokenized) > self.max_len_labels:
                        self.max_len_labels = len(All_Labels_Tokenized)
                else:
                    All_Labels_Tokenized = [0]
                    sent_gpt = "test"
                labs = row['label']
                gpt_sentence = "<bos> " + sentence_max_len.lower() + " <sep> " + labs.lower() + " <eos>"
                tokenized_labels = self.tokenizer.tokenize("<bos> " + labs + " <eos>")
                self.data[index] = {'input': input, 'label': tokenized_labels,
                                    'NoteMarginale': labs,
                                    'full_labels': row['label'], 'input_sentence': row['sentences']}
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
        return {
            'input_sentence':self.data[item]['input_sentence'],
            'full_labels':self.data[item]['full_labels'],
    }

    def iterexamples(self):
        for i, ex in self.data.items():
            yield i, ex
