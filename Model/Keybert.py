from keybert import KeyBERT
from eval import *
from tqdm import tqdm

class KeyBertModel():
	def __init__(self, argdict, datasets):
		super().__init__()

		self.argdict=argdict
		self.training_set, self.dev_set, self.test_set=datasets
		print(f"Training with {len(self.training_set)} exemples, and {len(self.training_set.index_unique_examples)} unique examples")
		print(f"Testing with {len(self.dev_set)} examples, and {len(self.dev_set.index_unique_examples)} unique examples")

		self.dico_mapping={'en':'english', 'fr':'french'}

		self.model=KeyBERT(model='distiluse-base-multilingual-cased-v2')
		# self.model=KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')


	def train_model(self):
		#Training:
		inputs=[]
		refs=[]
		hypos=[]
		# for i, exos in tqdm(self.dev_set.data.items()):
		# 	# print(exos)
		# 	inputs.append(exos['input_sentence'])
		# 	refs.append([rr.strip() for rr in exos['full_labels'].split(',')])
		# 	if self.kw_extractor is None:
		# 		try:
		# 			kw_extractor=yake.KeywordExtractor(lan=exos['language'])
		# 		except:
		# 			kw_extractor = yake.KeywordExtractor()
		# 	else:
		# 		kw_extractor=self.kw_extractor
		# 	gend=kw_extractor.extract_keywords(exos['input_sentence'])
		# 	hypos.append([kw[0] for kw in gend])
		#
		# score = evaluate(inputs, refs, hypos, '<unk>', tokenizer='split_nopunc')
		#
		# f10 = np.average(score['present_exact_f_score@10'])
		# r10 = np.average(score['absent_exact_recall@10'])
		# print(f"f1@10 present and r@10 absent for dev: {f10}, {r10}")

		for name, tt in self.test_set.items():
			print(f"running for {name} dataset")
			inputs = []
			refs = []
			hypos = []
			for i, exos in tqdm(tt.data.items()):
				inputs.append(exos['input_sentence'])
				refs.append([rr.strip() for rr in exos['full_labels'].split(',')])
				# ll=self.dico_mapping[exos['language']]
				gend=self.model.extract_keywords(exos['input_sentence'],keyphrase_ngram_range = (1,3),
												 stop_words = None, top_n = 10, nr_candidates = 20,
												 use_maxsum = True,
												 use_mmr = False,
												diversity = 0.7)
				hypos.append([kw[0] for kw in gend])
				# except:
				# 	hypos.append([])

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

