import pandas as pd

split='test'

df=pd.read_csv(f"../papyrus_f/{split}.tsv", index_col=0, sep='\t')
df2=pd.read_csv(f"../papyrus_e/{split}.tsv", index_col=0, sep='\t')
df=pd.concat([df, df2], ignore_index=True)

df.to_csv(f'{split}.tsv', sep='\t')