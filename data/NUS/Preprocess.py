import pandas as pd
import json
df=pd.DataFrame(columns=['sentences', 'label', 'title', 'language', 'index'])

Sentences=open("nus_test.src", "r").readlines()
Labels=open("nus_test.tgt", "r").readlines()

for i, (ss, ll) in enumerate(zip(Sentences, Labels)):
	ss=json.loads(ss)
	df.at[i, 'sentences']=ss['abstract']
	df.at[i, 'title']=ss['title']
	ll=json.loads(ll)
	df.at[i, 'label']=' , '.join(ll['keywords'])
	df.at[i, 'language']='en'
	df.at[i, 'index']=i

df.to_csv('test.tsv', sep='\t')