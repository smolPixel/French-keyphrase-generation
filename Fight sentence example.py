import pandas as pd


df=pd.read_csv("data/papyrus/dev.tsv", sep='\t', index_col=0)


indexes=list(df['index'])
language=list(df['language'])
len_language=[len(ll.split(',')) for ll in language]
for i, ll in enumerate(len_language):
	if ll>2:
		print(indexes[i])
		# print(i)
		print(language[i])
		print(ll)


index=1419
# index=2096

# print(df.loc[index])
print(df.at[index, 'title'])
print(df.at[index, 'sentences'])
print(df.at[index, 'label'])
print(df.at[index, 'language'])
print("---")

df=pd.read_csv("data/papyrus_f/train.tsv", sep='\t', index_col=0)
print(df.at['index'])