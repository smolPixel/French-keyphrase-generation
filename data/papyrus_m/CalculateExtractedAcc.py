import pandas as pd

df=pd.read_csv('checkAccLabelling.tsv', index_col=0, sep='\t')

labs=' '.join(list(df['checkLabs']))[:-2].split(' , ')
num_tot=len(labs)
num_err=len(['err' for lab in labs if lab=='N'])
print(num_err*100/num_tot)