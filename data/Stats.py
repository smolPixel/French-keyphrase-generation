"""Some stats on the test corpora"""
import pandas as pd
import numpy as np

datasets=['papyrus_f', 'papyrus_e', 'papyrus_m', 'papyrus', 'WikiNews',
		  'cacic57', 'wicc78', '110ptbnkp', 'pak2018', 'kp20k', 'Inspec',
		  'NUS', 'SemEval', 'krapivin']

for dataset in datasets:
	df=pd.read_csv(f"{dataset}/test.tsv", sep='\t', index_col=0)
	try:
		ll=list(df['language'])[0]
	except:
		ll='en'

	sents=list(df['sentences'])
	sents=[len(sent.split(' ')) for sent in sents]
	print(f"Dataset {dataset}. Language {ll} Number of test exos: {len(df)} of average length {np.mean(sents)}")