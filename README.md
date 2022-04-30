# French-keyphrase-generation

This repository includes the code for preprocessing and reproducing results associated with the submission to Neurips 2022 X (not submitted yet).

## Data
Data is available on X, and you can obtain the Kp20K data from Y. All data should be placed in the data folder. 

## Preprocessing.
Although not necessary for the papyrus, papyrus-f, papyrus-e, and papyrus-m tasks, all preprocessing scrips (scrapping, filtering, assigning languages to keyphrases, and other) are available in the preprocessing folder. For the Kp20K, simply run the preprocess.py file available in the Kp20K folder.

## Running BART for keyphrase generation. 
Once the data is correctly positionned in each folder, you can simply call python3 main.py --dataset X --num_epochs Y to train, save, and evaluate the model. 
