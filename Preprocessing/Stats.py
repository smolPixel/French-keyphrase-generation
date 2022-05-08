import pandas as pd
import numpy as np
from collections import Counter
splits=["train", "dev", "test"]
datasets=["papyrus_f", "papyrus_e", "papyrus_m", "papyrus"]#, "kp20k"]


df_og = pd.read_csv(f'data/papyrus_m/train.tsv', sep='\t', index_col=0)
# print(len(df_og))
# df = pd.DataFrame(columns=list(df_og))
print(len(df_og.groupby(['index'])))
new_index = 0
lengths=[]
for group in df_og.groupby(['index']):
	lengths.append(len(group[1]))

c=Counter(lengths)
print(c)
for dataset in datasets:
	for split in splits:
		df=pd.read_csv(f"data/{dataset}/{split}.tsv", index_col=0, sep='\t')
		lenOG=len(df)
		df=df.dropna()
		print(f"Dataset {dataset}, split {split}, length of {len(df)}, with {lenOG-len(df)} entries dropped")
	print('--')

print("Proportion of present/absent keyphrases")
for dataset in datasets:
	num_tot=0
	num_present=0
	num_absent=0
	df=pd.read_csv(f"data/{dataset}/train.tsv", index_col=0, sep='\t')
	df=df.dropna()
	for i, row in df.iterrows():
		abstract=row['sentences'].lower()
		keyphrases=row['label'].split(' , ')
		for kp in keyphrases:
			if kp.lower().strip() in abstract:
				num_present+=1
			else:
				num_absent+=1
			num_tot+=1
	print(f"The dataset {dataset} has {num_present/num_tot}% of present keyphrases and {num_absent/num_tot} keyphrases")
#
# dataset='kp20k'
# num_tot=0
# num_present=0
# num_absent=0
# df=pd.read_csv(f"data/{dataset}/train.tsv", index_col=0, sep='\t')
# df=df.dropna()
# for i, row in df.iterrows():
# 	abstract=row['sentences'].lower()
# 	keyphrases=row['label'].split(' , ')
# 	for kp in keyphrases:
# 		if kp.lower().strip() in abstract:
# 			num_present+=1
# 		else:
# 			num_absent+=1
# 		num_tot+=1
# print(f"The dataset {dataset} has {num_present/num_tot}% of present keyphrases and {num_absent/num_tot} keyphrases")


print("Average len of sentences and number of keyphrases")
for dataset in datasets:
	df = pd.read_csv(f"data/{dataset}/train.tsv", index_col=0, sep='\t')
	df=df.dropna()
	sents=list(df['sentences'])
	sents=[len(sent.strip().split(" ")) for sent in sents]
	labels=list(df['label'])
	labels_split_by_keyphrase=[kk.strip().split(",") for kk in labels]
	labels_split_by_keyphrase_by_space=[np.mean([len(ll.strip().split(' ')) for ll in entry]) for entry in labels_split_by_keyphrase]
	print(f"Average length of sentence  for {dataset}: {np.mean(sents)}\n\t number of keyphrases {np.mean([len(kk) for kk in labels_split_by_keyphrase])} \n\t len of keyphrases {np.mean(labels_split_by_keyphrase_by_space)}")

print("Number of doublons")
for dataset in datasets:
	nb_in_dev=0
	nb_in_test=0
	nb_test_dev=0
	dfTrain=pd.read_csv(f"data/{dataset}/train.tsv", index_col=0, sep='\t')
	dfDev=pd.read_csv(f"data/{dataset}/dev.tsv", index_col=0, sep='\t')
	titleDev=list(dfDev['sentences'])
	index_dev=list(dfDev['index'])
	dfTest=pd.read_csv(f"data/{dataset}/test.tsv", index_col=0, sep='\t')
	titleTest=list(dfTest['sentences'])

	for i, row in df.iterrows():
		title=row['sentences']
		if title in titleDev:
			nb_in_dev+=1
		if title in titleTest:
			print(row['title'])
			nb_in_test+=1

	for tt in titleDev:
		if tt in titleTest:
			nb_test_dev+=1
	print(f"The dataset {dataset} has {nb_in_dev} doublons dev/train, {nb_in_test} test/train, and {nb_test_dev} doublons test/dev")


print("Distribution of languages through the training corpus of papyrus")
df=pd.read_csv('data/papyrus/train.tsv', sep='\t', index_col=0)
list_language=[]
for i, row in df.iterrows():
	list_language.append(len(row['language'].split(',')))
	# fds
	# list_language.append(row['language'])

print(Counter(list_language))


