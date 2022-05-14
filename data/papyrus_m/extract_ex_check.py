import pandas as pd

df=pd.read_csv('train.tsv', index_col=0, sep='\t')
df=df.sample(n=100)
print(list(df))
enum=0
for i, row in df.iterrows():
	lang=row['language']
	sent=row['sentences']
	labs=row['label']
	print("----")
	print(enum)
	enum+=1
	print(lang)
	print(f"Sentence: {sent}")
	answer=input("Enter Y/N right language: ")
	df.at[i, 'checkSent']=answer
	checklabs=''
	for lab in labs.split(' , '):
		if lab.lower().strip() in sent.lower():
			checklabs+='PP , '
		else:
			print(f"keyphrase: {lab}")
			answer = input("Enter Y/N right language: ")
			checklabs+=f'{answer} , '
	df.at[i, 'checkLabs']=checklabs

df.to_csv('checkAccLabelling.tsv', sep='\t')