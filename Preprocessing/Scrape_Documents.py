import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

Max_index = 25000

def main():
	index=Max_index
	# April 7, 2022

	while index!=0:
		print(index)
		URL = f"https://papyrus.bib.umontreal.ca/xmlui/handle/1866/{index}?show=full"
		page = requests.get(URL)

		soup=BeautifulSoup(page.content, "html.parser")
		with open(f"Pages/{index}.html", "w", encoding='utf-8') as file:
			file.write(str(soup))
		time.sleep(0.5)
		index=index-1
		if index==24900:
			break

if __name__=='__main__':
	main()