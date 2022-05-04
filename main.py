import argparse, random, torch
import numpy as np
from util import *
from Model.BART import BARTModel
from Model.YAKE import YakeModel
from Model.BARTe import BARTeModel
from Model.BARTm import BARTMModel

def run_external_process(process):
	output, error = process.communicate()
	if process.returncode != 0:
		raise SystemError
	return output, error

def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

dicoAlgo={'bart':BARTModel,
		  'yake': YakeModel,
		  'barte':BARTeModel,
		  'bartm':BARTMModel}

def run(argdict):
	set_seed()
	datasets = initialize_datasets(argdict)
	model=dicoAlgo[argdict['algo']](argdict, datasets)
	model.train_model()
	if argdict['dataset'] not in ['kp20k']:
		model.generate_special_ex()
	# model.generate_from_dataset(split='train')
	# model.generate_from_dataset(split='dev')




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Using Barthez for keyphrase generation')
	#General arguments on training
	parser.add_argument('--dataset', type=str, default='Papyrus_f', help="dataset you want to run the process on. Includes Papyrus_f")
	parser.add_argument('--algo', type=str, default='bart', help='which algo do you want to run')
	parser.add_argument('--random_seed', type=int, default=7, help='Random seed ')
	parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, default=2, help='Batch Size')
	parser.add_argument('--accum_batch_size', type=int, default=64, help='Batch Size')
	parser.add_argument('--max_seq_length', type=int, default=1024, help='max length, 0 if no max length')


	parser.add_argument('--short_eval', action='store_true', help='Cut evaluation time for debugging purposes')




	args = parser.parse_args()
	argsdict = args.__dict__

	if argsdict['dataset'].lower()=='papyrus_f':
		argsdict['language']='fr'
	elif argsdict['dataset'].lower() in ['kp20k', 'papyrus_e', 'kp20kpapyruse']:
		argsdict['language']='en'
	elif argsdict['dataset'].lower() in ['papyrus_m', 'papyrus']:
		argsdict['language']='mu'
	else:
		raise ValueError("Unrecognized dataset")

	run(argsdict)