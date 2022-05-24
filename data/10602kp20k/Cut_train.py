import pandas as pd

df=pd.read_csv('train.tsv', index_col=0, sep='\t')
df=df.sample(n=10602)


df.to_csv('train.tsv', sep='\t')