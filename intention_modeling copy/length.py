import csv
import pandas as pd
def main():
	f = open('not_in_sent.txt', 'w+')
	# open the sentiment dataset 
	df = pd.read_csv('cleaned_data_train.tsv', sep="\t")
	# create a lsit to put all the tweet IDs from sentiment 
	sent = []
	for i in range(df.shape[0]):
		sent.append(str(df['Tweet ID'][i]))
	
	with open("Vulgar_Functions_Dataset.csv") as f2:
		read = csv.reader(f2, delimiter= ',')
		data = [r for r in read]
	
	ex = []
	for i in range(1, len(data)):

		ex.append(str(data[i][0]))
	'''with open('train_set.txt', 'r') as f:
		train_tweets = f.read().split("\n")
	while '' in train_tweets:
		train_tweets.remove("")
	print(len(train_tweets))'''
	print(len(set(ex)), len(sent))
	vulgar = list(set(ex))		# unique data for the training set of vulgar intentions
	sent = list(set(sent))		# unique data for the training set of sentiment 
	new = []					# the vulgar data that is in sentiment 
	hmm = []					# the vulgar data that is not in sentiment 
	for i in range(len(sent)):
		if sent[i] in vulgar:
			new.append(sent[i])
		else:
			hmm.append(sent[i])
	print(len(new))
	for i in range(len(hmm)):
		f.write(hmm[i] + '\n')



main()

