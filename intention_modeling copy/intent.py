import csv
import pandas as pd
import numpy as np
def main():
	with open('train_set.txt', 'r') as f:
		train_tweets = f.read().split("\n")
	while '' in train_tweets:
		train_tweets.remove("")
	
	df = pd.read_csv('cleaned_data_train.tsv', sep="\t")
	sent = []
	for i in range(df.shape[0]):
		a = []
		a.append(str(df['Tweet ID'][i]))
		a.append(str(df['(1) Very Negative'][i]))
		a.append(str(df['(2) Negative'][i]))
		a.append(str(df['(3) Neutral'][i]))
		a.append(str(df['(4) Positive'][i]))
		a.append(str(df['(5) Strongly Positive'][i]))
		a.append(str(df['NA'][i]))
		a.append(str(df['Majority'][i]))
		a.append(str(df['num_vulgar'][i]))
		sent.append(a)

	for line in range(1, len(train_tweets)):

		line_array = train_tweets[line].split('\t')

		for i in range(len(sent)):
			if line_array[1] in sent[i]:
				line_array.append(sent[i][1])
				line_array.append(sent[i][2])
				line_array.append(sent[i][3])
				line_array.append(sent[i][4])
				line_array.append(sent[i][5])
				line_array.append(sent[i][6])
				line_array.append(sent[i][7])
				line_array.append(sent[i][8])
				break

		if len(line_array) == 14:
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')
				line_array.append('0')

		print(len(line_array))

main()