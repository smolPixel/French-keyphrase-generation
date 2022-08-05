import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('train.tsv', sep='\t', index_col=0)

print(list(df))

ss=list(df['sentences'])
ss=[len(s) for s in ss]

sorted_ss=np.argsort(ss)
print(sorted_ss)
sents=np.array(list(df['sentences']))
print(sents[sorted_ss[:5]])


plt.hist(ss, bins=100)
plt.savefig('test.png')