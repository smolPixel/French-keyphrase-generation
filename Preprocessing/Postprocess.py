import pandas as pd

datasets=['papyrus_f', 'papyrus_m', 'papyrus_e', 'papyrus']


for dataset in datasets:
	dfTrain=pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', index_col=0).dropna()
	dfDev=pd.read_csv(f'data/{dataset}/dev.tsv', sep='\t', index_col=0).dropna()
	dfTest=pd.read_csv(f'data/{dataset}/test.tsv', sep='\t', index_col=0).dropna()

	dfTrain=dfTrain[~dfTrain['sentences'].isin(dfTest['sentences'])]
	dfTrain=dfTrain[~dfTrain['sentences'].isin(dfDev['sentences'])]
	dfDev=dfDev[~dfDev['sentences'].isin(dfTest['sentences'])]

	dfTrain.to_csv(f'data/{dataset}/train.tsv', sep='\t')
	dfDev.to_csv(f'data/{dataset}/dev.tsv', sep='\t')
	dfTest.to_csv(f'data/{dataset}/test.tsv', sep='\t')
