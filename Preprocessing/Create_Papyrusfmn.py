"""Creates papyrus e, f, and Papyrus from papyrus-m"""

import pandas as pd

splits=['train', 'dev', 'test']

for split in splits:
	df_og=pd.read_csv(f'data/papyrus_m/{split}.tsv', sep='\t', index_col=0)
	fr_df=pd.DataFrame(columns=list(df_og))

	new_index=0
	for i, row in df_og.iterrows():
		if row['index']==12533:
			print(row['title'])
			print(row['sentences'])
			print(row['label'])
		if row['language']=='fr':
			fr_df.loc[new_index]=row
			new_index+=1

	fr_df.to_csv(f"data/papyrus_f/{split}.tsv", sep='\t')

	df_og=pd.read_csv(f'data/papyrus_m/{split}.tsv', sep='\t', index_col=0)
	fr_df=pd.DataFrame(columns=list(df_og))

	new_index=0
	for i, row in df_og.iterrows():
		if row['language']=='en':
			fr_df.loc[new_index]=row
			new_index+=1

	fr_df.to_csv(f"data/papyrus_e/{split}.tsv", sep='\t')


	df_og=pd.read_csv(f'data/papyrus_m/{split}.tsv', sep='\t', index_col=0)
	df=pd.DataFrame(columns=list(df_og))

	new_index=0
	for group in df_og.groupby(['index']):
		sentence=[]
		label=[]
		languages=[]
		title=""
		index=0
		for index, serie in group[1].iterrows():
			sentence.append(serie.at['sentences'])
			languages.append(serie.at['language'])
			if not pd.isna(serie['label']):
				label.append(serie['label'])
			title=serie["title"]
			index=serie["index"]
		df.at[new_index, 'sentences']=" ".join(sentence)
		df.at[new_index, 'language']=" , ".join(languages)
		df.at[new_index, 'label']=" , ".join(label)
		df.at[new_index, 'title']=title
		df.at[new_index, "index"]=index
		new_index+=1
		# if new_index==10:
		# 	break

	# for i, row in df_og.iterrows():
	# 	if row['language']=='fr':
	# 		fr_df.loc[new_index]=row
	# 		new_index+=1

	df.to_csv(f"data/papyrus/{split}.tsv", sep='\t')