# French-keyphrase-generation

This repository includes the code for preprocessing and reproducing results associated with the submission to Neurips 2022 X (not submitted yet).

## Data
Data is available on X, and you can obtain the Kp20K data from Y. All data should be placed in the data folder. 

## Preprocessing.
Although not necessary for the papyrus, papyrus-f, papyrus-e, and papyrus-m tasks, 
all preprocessing scrips (scrapping, filtering, assigning languages to keyphrases, a
nd other) are available in the preprocessing folder. For the Kp20K, simply run the
 preprocess.py file available in the Kp20K folder. If you want to recollect the entirety of the 
 corpus (or create an updated version). You should:
 
 1. Go to Papyrus (https://papyrus.bib.umontreal.ca) and figure out what's the latest document
 updated and it's index. Pages are of the format https://papyrus.bib.umontreal.ca/xmlui/handle/1866/INDEX.
 2. In the Preprocessing/Scrape_Documents.py, change the Max_index variable with this index.

NOTE: The process of fasttext uses lid.176.bin, which you can download from https://fasttext.cc/docs/en/language-identification.html
and place in a folder called Models inside the Preprocessing folder (or wherever you want and change the path in
Process.py)

## Running BART for keyphrase generation. 
Once the data is correctly positionned in each folder, you can simply call python3 main.py --dataset X --num_epochs Y to train, save, and evaluate the model. 
