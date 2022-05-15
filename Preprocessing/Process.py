import pandas as pd
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
import fasttext
from langdetect import detect, detect_langs

nb_ex=26508
random.seed(47)
np.random.seed(47)
indexes=np.arange(nb_ex)
np.random.shuffle(indexes)
train_indexes=indexes[:int(0.7*nb_ex)]
dev_indexes=indexes[int(0.7*nb_ex):int(0.8*nb_ex)]
test_indexes=indexes[int(0.8*nb_ex):]



f = open("dataset.jsonl", "r")
dfTrain=pd.DataFrame(columns=['title', 'sentences', 'label'])
dfDev=pd.DataFrame(columns=['title', 'sentences', 'label'])
dfTest=pd.DataFrame(columns=['title', 'sentences', 'label'])

model=fasttext.load_model('Models/lid.176.bin')

def add_to_df(df, dico_temp, line):
	for key, value in dico_temp.items():
		lendf=len(df)
		df.at[lendf, 'title']=line['title'][0]
		df.at[lendf, 'sentences']=value[0].replace('\t', ' ').replace('\n', ' ')
		df.at[lendf, 'label']= ' , '.join(value[1])
		df.at[lendf, 'language']= key
		df.at[lendf, 'index']=line['index']

total_not_found=0
total_keyphrases=0
prop_weirdos=0
new_index=0
#For doublons
list_all_abastracts=[]
while True:
	if new_index%1000==0:
		print(new_index)
	line=f.readline()
	if line=="":
		break
	try:
		line=json.loads(line)
	except:
		#We reached the end of the file
		print(df)

	abs=line['abstract']
	keyphrases=line['keyphrases']
	index=line['index']
	if index==12533:
		print(line)
	dicoTemp={}
	#Identify languages of abstract
	languages_in_text=[]
	for text in abs:
		try:
			language=detect_langs(text)
		except:
			if index==2764:
				#Erroneously tagged abstract
				continue
			language='un'
		language=str(language[0])[:2]
		#If an abstract of this language already exists
		if language in languages_in_text:
			#Take the longer one
			if len(text)>len(dicoTemp[language][0]):
				dicoTemp[language]=(text, [])
				continue
		dicoTemp[language]=(text, [])
		languages_in_text.append(language)
	languages_in_text=list(dicoTemp.keys())
	if "unknown" in languages_in_text:
		print(abs)
	#Get language codes, langdetect is not great on short phrases so we do not care about the exact probabilities anyway
	for kp in keyphrases:
		total_keyphrases+=1
		foundLanguage=False

		if '(UMI' in kp or '[JEL' in kp:
			prop_weirdos+=1
			# print(kp[0])
			continue

		#Check if there's only one language -> all kp are of this language
		if len(languages_in_text)==1:
			dicoTemp[languages_in_text[0]][1].append(kp)
			continue

		languages_of_kp=[]
		#If it's word for word in an abstract, it belongs to that language
		for lang, item in dicoTemp.items():
			abstract, _=item
			if kp.lower() in abstract.lower():
				languages_of_kp.append(lang)
				foundLanguage=True
		if foundLanguage:
			for ll in languages_of_kp:
				dicoTemp[ll][1].append(kp)
			continue

		languages_fasttext=model.predict(kp, k=15)
		# print(kp[0])
		# print(kp)
		languages_kp=[ll[-2:] for ll in languages_fasttext[0]]
		for ll in languages_kp:
			if ll in languages_in_text:
				dicoTemp[ll][1].append(kp)
				foundLanguage=True
				break

		if not foundLanguage:
			# print(kp[0])
			# print(languages_kp)
			total_not_found+=1
			for ll in languages_in_text:
				dicoTemp[ll][1].append(kp)


	# fds
	#Putting in appropriate dataframe:
	if new_index in train_indexes:
		add_to_df(dfTrain, dicoTemp, line)
	elif new_index in dev_indexes:
		add_to_df(dfDev, dicoTemp, line)
	elif new_index in test_indexes:
		add_to_df(dfTest, dicoTemp, line)
	new_index+=1



dfTrain.to_csv('data/papyrus_m/train.tsv', sep='\t')
dfDev.to_csv('data/papyrus_m/dev.tsv', sep='\t')
dfTest.to_csv('data/papyrus_m/test.tsv', sep='\t')
