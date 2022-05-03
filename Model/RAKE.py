# Implementation of RAKE - Rapid Automtic Keyword Exraction algorithm
# as described in:
# Rose, S., D. Engel, N. Cramer, and W. Cowley (2010).
# Automatic keyword extraction from indi-vidual documents.
# In M. W. Berry and J. Kogan (Eds.), Text Mining: Applications and Theory.unknown: John Wiley and Sons, Ltd.

import re
import operator

debug = False
test = True

from eval import *
from tqdm import tqdm

class RakeModel():
	def __init__(self, argdict, datasets):
		super().__init__()

		self.argdict=argdict
		self.training_set, self.dev_set, self.test_set=datasets
		print(f"Training with {len(self.training_set)} exemples, and {len(self.training_set.index_unique_examples)} unique examples")
		print(f"Testing with {len(self.dev_set)} examples, and {len(self.dev_set.index_unique_examples)} unique examples")

		if argdict['language']=='en':
			self.kw_extractor = yake.KeywordExtractor()
		elif argdict['language']=='fr':
			self.kw_extractor = yake.KeywordExtractor(lan="fr")
		elif argdict['language']=='mu':
			#will be defined later
			self.kw_extractor=None
		else:
			raise ValueError("Unrecognized language")


	def train_model(self):
		#Training:
		inputs=[]
		refs=[]
		hypos=[]
		for i, exos in tqdm(self.dev_set.data.items()):
			# print(exos)
			inputs.append(exos['input_sentence'])
			refs.append([rr.strip() for rr in exos['full_labels'].split(',')])
			if self.kw_extractor is None:
				try:
					kw_extractor=yake.KeywordExtractor(lan=exos['language'])
				except:
					kw_extractor = yake.KeywordExtractor()
			else:
				kw_extractor=self.kw_extractor
			gend=kw_extractor.extract_keywords(exos['input_sentence'])
			hypos.append([kw[0] for kw in gend])

		score = evaluate(inputs, refs, hypos, '<unk>', tokenizer='split_nopunc')
		f10 = np.average(score['present_exact_f_score@10'])
		r10 = np.average(score['absent_exact_recall@10'])
		print(f"f1@10 present and r@10 absent for dev: {f10}, {r10}")

		inputs = []
		refs = []
		hypos = []
		for i, exos in tqdm(self.test_set['test'].data.items()):
			# print(exos)
			inputs.append(exos['input_sentence'])
			refs.append([rr.strip() for rr in exos['full_labels'].split(',')])
			if self.kw_extractor is None:
				try:
					kw_extractor=yake.KeywordExtractor(lan=exos['language'])
				except:
					kw_extractor = yake.KeywordExtractor()
			else:
				kw_extractor=self.kw_extractor
			gend=kw_extractor.extract_keywords(exos['input_sentence'])
			hypos.append([kw[0] for kw in gend])

		score = evaluate(inputs, refs, hypos, '<unk>', tokenizer='split_nopunc')
		f10 = np.average(score['present_exact_f_score@10'])
		r10 = np.average(score['absent_exact_recall@10'])
		print(f"f1@10 present and r@10 absent for test: {f10}, {r10}")

	def generate_special_ex(self):
		print(f"Generation for example 26XXX")
		dataset = self.training_set
		inputs = dataset.abstract_for_ex
		refs = dataset.label_for_ex
		hypos = []

		for abstract in inputs:
			gend=self.kw_extractor.extract_keywords(abstract)
			hypos.append([kw[0] for kw in gend])
		# hypos.append(gend)


		# print(inputs, hypos, refs)

		for ii, hh, rr in zip(inputs, hypos, refs):
			print(f"Input : {ii} \n "
				  f"Note Marginale: {rr} \n"
				  f"Note Générée: {hh}")
			print("------------------")

		print("GENERATION FOR THE DEV SET")
		print(f"Generation for example 24192")
		dataset = self.dev_set
		inputs = dataset.abstract_for_ex
		refs = dataset.label_for_ex
		hypos = []

		for abstract in inputs:
			gend=self.kw_extractor.extract_keywords(abstract)
			hypos.append([kw[0] for kw in gend])
		for ii, hh, rr in zip(inputs, hypos, refs):
			print(f"Input : {ii} \n "
				  f"Note Marginale: {rr} \n"
				  f"Note Générée: {hh}")
			print("------------------")


def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words


def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences


def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern


def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list


def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        #if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  #orig.
            #word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
    #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score


def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates


class Rake(object):
    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)

    def run(self, text):
        sentence_list = split_sentences(text)

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)

        sorted_keywords = sorted(keyword_candidates.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords