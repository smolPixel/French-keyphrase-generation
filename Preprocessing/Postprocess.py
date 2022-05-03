import pandas as pd

datasets=['papyrus_f', 'papyrus_m', 'papyrus_e', 'papyrus']


for dataset in datasets:
	dfTrain=pd.read_csv(f'../data/{dataset}/train.tsv', sep='\t', index_col=0)
	dfDev=pd.read_csv(f'../data/{dataset}/dev.tsv', sep='\t', index_col=0)
	dfTest=pd.read_csv(f'../data/{dataset}/test.tsv', sep='\t', index_col=0)

	print(len(dfTrain))
	dfTrain=dfTrain[~dfTrain['sentences'].isin(dfTest['sentences'])]
	print(len(dfTrain))