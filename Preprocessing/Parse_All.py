from bs4 import BeautifulSoup
from langdetect import detect, detect_langs
from Scrape_Documents import Max_index
import json

index_start=Max_index
error_pages=0
no_abstr_or_kp=0
# index_start=12533
# file=open("dataset_notFiltered.jsonl", "r")
file=open("dataset.jsonl", "w")

dico_nb_abstracts={}
languages_abstracts={}

num_franco=0
num_multilingue=0
i=0
while True:
	i+=1
	# print(i)
	# print(index_start)
	if i%1000==0:
		print(i)
	authors = []
	abstracts = []
	title = []
	keywords = []
	with open(f"Pages/{index_start}.html", encoding='utf-8') as fp:
		soup = BeautifulSoup(fp, 'html.parser')
		for tag in soup.find_all("meta"):
			if tag.get("name", None) == "DCTERMS.abstract":
				abstracts.append(tag.get("content", None))
			elif tag.get("name", None)=="citation_title":
				title.append(tag.get("content", None))
			elif tag.get("name", None)== "DC.subject":
				keywords.append(tag.get("content", None))
		"""First putting everything in json format for faster access"""
		dict = {'index': index_start, 'title': title, 'abstract': abstracts,
				'keyphrases':keywords}
		json.dump(dict, file, ensure_ascii=False)
		file.write('\n')
		index_start-=1
		continue


# print(f"Number of error pages: {error_pages}, Number with no abstracts or keyphrases: {no_abstr_or_kp}")
# print(f"Distribution of the number of abstracts: {dico_nb_abstracts}")
# print(f"Distribution of the languages: {languages_abstracts}")