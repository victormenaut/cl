import nltk 
import sys 
import numpy 
from nltk.corpus import brown

def count_tag(corpus):
	count_tag = {}
	for word,tag in corpus:

		if word not in count_tag:
			count_tag[word] = {} 
		if tag not in count_tag[word]:
			count_tag[word][tag] = 1
		else:
			count_tag[word][tag] += 1
	return MLE_word_tag(count_tag)

	
def MLE_word_tag(dicts):
	count_tag = dicts
	total_words = len(count_tag)
	prob_word = {} 
	for word in count_tag:
		count = 0
		total_tag = 0
		for tag in count_tag[word]:		#value of the tag count 
			total_tag += count_tag[word][tag]
		for tag in count_tag[word]:
			count = count_tag[word][tag]
			prob_tag = count/total_tag
			count_tag[word][tag] = prob_tag
		prob_word[word] = count/total_words
	prob_tag = count_tag

	return prob_word, prob_tag

def forwardprobs(observe, initial, trans, emis, numstates, observe_indices):
	forwardmatrix = numpy.zeros((numstates, len(observe)))
	obs_index = observe_indices[observe[0]]
	for s in range(numstates):
		forwardmatrix[s,0] = initial[s]*emis[s, obs_index]

	for t in range(1, len(observe)):
		obs_index = observe_indices[observe[t]]
		for s in range(numstates):
			forwardmatrix[s, t] = emis[s,obs_index] *sum([forwardmatrix[s2, t]* \
			trans[s2, s] for s2 in range(numstates)])

	return forwardmatrix


def backwardprobs(observe, trans, emis, numstates, observe_indices):

	backwardmatrix = numpy.zeros((numstates, len(observe)))
	for s in range(numstates):
		backwardmatrix[s, len(observe)-1] = 1.0 

	for t in range(len(observe)-2, -1, -1):
		obs_index = observe_indices[observe[t+1]]
		for s in range(numstates):
			backwardmatrix[s,t] = sum([trans[s, s2]* emis[s2, obs_index]*backwardmatrix[s2, t+1]\
				for s2 in range(numstates)])

	return backwardmatrix


def test_alphabeta():
    observations = [3,1,3]
    trans = numpy.matrix("0.7 0.3; 0.4 0.6")
    emis = numpy.matrix("0.2 0.4 0.4; 0.5 0.4 0.1")
    initialprob = numpy.array([0.8, 0.2])
    numstates = 2
    obs_indices = { 1 : 0, 2 : 1, 3: 2}

    print("FORWARD")
    print(forwardprobs(observations, initialprob, trans, emis, numstates, obs_indices))
    print("\n")
    
    print('BACKWARD')
    print(backwardprobs(observations, trans, emis, numstates, obs_indices))
    print("\n")

		



def unknown_tag(word,prev, dicts, tag):
	a = word
	b = dicts
	c = tag
	d = prev
	post = []
	for i in range(len(tag)-1):
		if prev in tag[i]:
			post.append(tag[i+1])
	post_tag = {}
	for j in range(len(post)):
		tags = post[j]
		if tags[1] not in post_tag:
			post_tag[tags[1]] = 1
		else:
			post_tag[tags[1]] += 1
	return (word + "/" +  max(post_tag, key =post_tag.get ))
			
			
def main():
	tag = brown.tagged_words()
	prob = count_tag(tag)
	s = "this tagger is stupid"
	a = s.split()
	for i in range (len(a)):
		if a[i] in prob[1]:
			word = prob[1][a[i]]

			if len(word) > 1:
				mle = 0
				for key,value in word.items():
					if word[key] > mle:
						mle = word[key]
						tagged = key
					else:
						mle = mle

			else:
				for key, value in word.items():
					mle = word[key]
					tagged = key

			print(a[i] + '/' + tagged, end = " ")
		else:
			b = unknown_tag(a[i], a[i-1],prob[1], tag)
			print (b, end = " ")
		
main()
