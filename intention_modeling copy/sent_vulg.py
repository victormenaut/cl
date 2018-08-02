import csv
import pandas as pd
def main():
	f = open('new_sent.txt', 'w+')
	f2 = open('new_vulgar.txt', 'w+')
	df = pd.read_csv('cleaned_data_train.tsv', sep="\t")
	sent = []
	for i in range(df.shape[0]):
		sent.append(str(int(df['Tweet ID'][i])))

	bc = pd.read_csv('Vulgar_Functions_Dataset.csv', sep=",")
	vulg = []
	for i in range(bc.shape[0]):	
		vulg.append(str(int(bc['tweet_id'][i]))[:-4])
	

	new = []
	hmm = []
	for i in range(len(sent)):
		if sent[i][:-4] not in vulg:
			
			new.append(sent[i])
		else:
			hmm.append(sent[i])
	#  new is the sentiment tweets that are not found in the vulgar intentions there are 43 

	
main()