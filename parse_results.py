import numpy as np

bartf={'wicc78':{'f110p':[], 'r10a':[]},
	   '110ptbnkp':{'f110p':[], 'r10a':[]},
	   'wikinews':{'f110p':[], 'r10a':[]},
	   'cacic57':{'f110p':[], 'r10a':[]},
	   'pak2018':{'f110p':[], 'r10a':[]},
	   'papyruse':{'p5p':[], 'r5p':[], 'f15p':[], 'p10p':[], 'r10p':[], 'f110p':[],
					'p10a':[], 'r10a':[],'f110a':[]},
	   'papyrusf':{'p5p':[], 'r5p':[], 'f15p':[], 'p10p':[], 'r10p':[], 'f110p':[],
					'p10a':[], 'r10a':[],'f110a':[]},
	   'papyrusm':{'p5p':[], 'r5p':[], 'f15p':[], 'p10p':[], 'r10p':[], 'f110p':[],
					'p10a':[], 'r10a':[],'f110a':[]},
	   'papyrus':{'p5p':[], 'r5p':[], 'f15p':[], 'p10p':[], 'r10p':[], 'f110p':[],
					'p10a':[], 'r10a':[],'f110a':[]},
	   'kp20k':{'p5p':[], 'r5p':[], 'f15p':[], 'p10p':[], 'r10p':[], 'f110p':[],
					'p10a':[], 'r10a':[],'f110a':[]},
	   'semeval':{'f110p':[], 'r10a':[]},
	   'krapivin': {'f110p': [], 'r10a': []},
	   'nus': {'f110p': [], 'r10a': []},
	   'inspec': {'f110p': [], 'r10a': []},
	   }

seeds=[42,43,44]
datasets=['wicc78', '110ptbnkp', 'wikinews', 'cacic57', 'pak2018', 'papyruse', 'papyrusf', 'papyrusm', 'papyrus', 'kp20k', 'semeval', 'krapivin', 'nus', 'inspec']

for seed in seeds:
	file=open(f"Results/t5_papyrus_f_seed_{seed}.txt").read()
	# file=open(f"seed{seed}.txt").read()
	file=file.split('Percent of absent generated keyphrases')[:-1]
	# file=file.split('Running test for')[1:]
	short_eval=False
	for dataset, results in zip(datasets, file):
		if len(bartf[dataset])==2:
			short_eval=True
		else:
			short_eval=False
		for line in results.split('\n'):
			if 'prec@5 present Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['p5p'].append(results)
			elif 'rec@5 present Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['r5p'].append(results)
			elif 'f1@5 present Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['f15p'].append(results)
			elif 'prec@10 present Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['p10p'].append(results)
			elif 'rec@10 present Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['r10p'].append(results)
			elif 'f1@10 present Test' in line:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['f110p'].append(results)
			elif 'prec@10 absent Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['p10a'].append(results)
			elif 'rec@10 absent Test' in line and not 'prec@10 absent Test' in line:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['r10a'].append(results)
			elif 'f1@10 absent Test' in line and not short_eval:
				results=float(line.split(' : ')[-1].strip())
				bartf[dataset]['f110a'].append(results)

for key, item in bartf.items():
	print('----------')
	print(key)
	print('-------------')
	for metric, result in item.items():
		print(f"{metric}: {np.mean(result)*100}, {np.std(result)*100} ({len(result)} results)")