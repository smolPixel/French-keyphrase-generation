import pandas as pd
import numpy as np

df=pd.read_csv('train.tsv', sep='\t', index_col=0)


labs=list(df['label'])
labs=[kp for ll in labs for kp in ll.split(' , ') ]
np.random.shuffle(labs)
checked=0
answers=[]
for ll in labs:
	print(ll)
	inp=input("Could this be considered an english keyphrase: ")
	answers.append(int(inp))
	checked+=1
	if checked%25==0:
		print(checked)
	if checked==100:
		print(sum(answers))
		fds