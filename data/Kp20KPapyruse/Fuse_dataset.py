import pandas as pd

df=pd.read_csv("../kp20k/train.tsv", index_col=0, sep='\t')
df=df.drop('wordExtracted', 1)
df['language']='en'
df['index']=0
df2=pd.read_csv("../papyrus_e/train.tsv", index_col=0, sep='\t')


df=pd.concat([df, df2], ignore_index=True)

df.to_csv('train.tsv', sep='\t')