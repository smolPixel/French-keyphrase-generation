import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('train.tsv', sep='\t', index_col=0)

print(list(df))

ss=list(df['sentences'])
ss=[len(s.split(' ')) for s in ss]

plt.hist(ss, bins=100)
plt.savefig('test.png')