import pandas as pd
import os

df=pd.DataFrame(columns=['title', 'sentences', 'label', 'language',	'index'])


index=0
for file in os.listdir('docsutf8'):
	nameFile=file
	index_file=nameFile.split('.')[0]
	text=open(f'docsutf8/{index_file}.txt', 'r').read().strip()#replace('.\n', '. ').replace('\n', ' ').strip()
	text=text.split('\n')
	title=text[0]
	text=' '.join(text[1:]).strip()
	keys=open(f'keys/{index_file}.key', 'r').read()
	df.at[index, 'title']=title
	df.at[index, 'sentences']=text
	df.at[index, 'label']= ' , '.join(keys.strip().split('\n'))
	df.at[index, 'language']='pl'
	df.at[index, 'index']=index
	index+=1


df.to_csv('test.tsv', sep='\t')