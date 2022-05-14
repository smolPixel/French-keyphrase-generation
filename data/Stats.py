"""Some stats on the test corpora"""
import pandas as pd
import numpy as np

datasets=['papyrus_f', 'papyrus_e', 'papyrus_m', 'papyrus', 'WikiNews',
		  'cacic57', 'wicc78', '110ptbnkp', 'pak2018', 'kp20k', 'Inspec',
		  'NUS', 'SemEval', 'krapivin']



for dataset in datasets:
	df=pd.read_csv(f"{dataset}/test.tsv", sep='\t', index_col=0).dropna()
	try:
		ll=list(df['language'])[0]
	except:
		ll='en'

	sents=list(df['sentences'])
	sents=[len(sent.split(' ')) for sent in sents]
	labels = list(df['label'])
	labels_split_by_keyphrase = [kk.strip().split(",") for kk in labels]
	labels_split_by_keyphrase_by_space = [np.mean([len(ll.strip().split(' ')) for ll in entry]) for entry in
										  labels_split_by_keyphrase]
	num_tot = 0
	num_present = 0
	num_absent = 0
	num_absent_broken=0
	for i, row in df.iterrows():
		abstract = row['sentences'].lower()
		keyphrases = row['label'].split(' , ')
		for kp in keyphrases:
			if kp.lower().strip() in abstract:
				num_present += 1
			else:
				num_absent += 1
			num_tot += 1

		#Absent broken
		for kp in keyphrases:
			words_individual=kp.split(" ")
			print(words_individual)
			fds
	# print(
	# 	f"Average length of sentence  for {dataset}: {np.mean(sents)}\n\t number of keyphrases {np.mean([len(kk) for kk in labels_split_by_keyphrase])} \n\t len of keyphrases {np.mean(labels_split_by_keyphrase_by_space)}")
	print(f"Dataset {dataset}. Language {ll} Number of test exos: {len(df)} of average length {np.mean(sents)}\n\t"
		  f"Number of keyphrases {np.mean([len(kk) for kk in labels_split_by_keyphrase])} len of keyphrases {np.mean(labels_split_by_keyphrase_by_space)} and {num_present / num_tot}% of present keyphrases")