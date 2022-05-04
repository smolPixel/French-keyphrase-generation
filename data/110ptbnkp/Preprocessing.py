import pandas as pd
import os

df=pd.DataFrame(columns=['title', 'sentences', 'label', 'language',	'index'])


index=0
for file in os.listdir('docsutf8'):
	nameFile=file
	index_file='.'.join(nameFile.split('.')[:-1])
	text=open(f'docsutf8/{index_file}.txt', 'r').read().strip()#replace('.\n', '. ').replace('\n', ' ').strip()
	keys=open(f'keys/{index_file}.key', 'r').read()
	df.at[index, 'title']="No title given"
	df.at[index, 'sentences']=text.replace('\t', ' ')
	df.at[index, 'label']= ' , '.join(keys.strip().split('\n')).replace('\t', ' ')
	df.at[index, 'language']='pt'
	df.at[index, 'index']=index_file.replace('\t', ' ')
	index+=1


df.to_csv('test.tsv', sep='\t')