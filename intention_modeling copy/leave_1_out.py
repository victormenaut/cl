import sklearn.linear_model
import numpy as np
import string
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut
#from sklearn import cross_validation

def instantiate_clusters():
    # fpath = "brown_clusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt"
    fpath = "brown_clusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c320-freq1.txt"
    # fpath = "brown_clusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c1000-freq1.txt"
    # fpath = "brown_clusters/brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt"

    all_clusters = {}
    with open(fpath, 'r') as f:
        for line in f:
            if line != '':
                spline = line.split("\t")
                word = spline[1]
                cluster = spline[0]

                if word not in all_clusters:
                    all_clusters[word] = cluster

    return all_clusters

def cluster_one_hot_dict(train_vals):
    cluster_one_hot_mapping = {}
    counter = 0
    for val in train_vals:
        p_val = val[0]
        n_val = val[1]
        if p_val in cluster_one_hot_mapping:
            pass
        else:
            cluster_one_hot_mapping[p_val] = counter
            counter += 1

        if n_val in cluster_one_hot_mapping:
            pass
        else:
            cluster_one_hot_mapping[n_val] = counter
            counter += 1

    return cluster_one_hot_mapping

def cluster_one_hot_dict_whole(train_vals):
    cluster_one_hot_mapping = {}
    counter = 0
    for val in train_vals:
        for cluster in val:
            if cluster in cluster_one_hot_mapping:
                pass
            else:
                cluster_one_hot_mapping[cluster] = counter
                counter += 1

    return cluster_one_hot_mapping

def one_hot_cluster(val, cluster_one_hot_mapping):
    p_one_hot = np.zeros(len(cluster_one_hot_mapping))
    n_one_hot = np.zeros(len(cluster_one_hot_mapping))

    p_val = val[0]
    n_val = val[1]
    if p_val in cluster_one_hot_mapping:
        p_one_hot[cluster_one_hot_mapping[p_val]] += 1
    else:
        p_one_hot[cluster_one_hot_mapping["0000"]] += 1

    if n_val in cluster_one_hot_mapping:
        n_one_hot[cluster_one_hot_mapping[n_val]] += 1
    else:
        n_one_hot[cluster_one_hot_mapping["0000"]] += 1

    return p_one_hot, n_one_hot

def one_hot_cluster_whole(val, cluster_one_hot_mapping):
    one_hot = np.zeros(len(cluster_one_hot_mapping))

    for clust in val:
        if clust in cluster_one_hot_mapping:
            one_hot[cluster_one_hot_mapping[clust]] += 1

    return one_hot


def get_brown_feature_context(tweet, index, cluster_dict):
    words = tweet.split()

    if index > 0:
        if words[index - 1] in cluster_dict:
            p_word = cluster_dict[words[index - 1]]
        else:
            p_word = "0000"
    else:
        p_word = "0"

    if index != len(words) - 1:
        if words[index + 1] in cluster_dict:
            next_word = cluster_dict[words[index + 1]]
        else:
            next_word = "0000"
    else:
        next_word = "1"

    return p_word, next_word

def get_brown_feature_whole(tweet, cluster_dict):
    words = tweet.split()
    return_me = []

    for word in words:
        if word in cluster_dict:
            return_me.append(cluster_dict[word])

    return return_me


def check_pronoun(tweet, index):
    pronouns = ["i", "me", "you", "he", "she", "they", "him", "her", "them", "we", "us"]
    pronouns = set(pronouns)
    words = tweet.split()

    # Check three words before
    # if index - 3 >= 0:
    #     if words[index - 3].lower().strip() in pronouns:
    #         return True
    # if index - 2 >= 0:
    #     if words[index - 2].lower().strip() in pronouns:
    #         return True
    if index - 1 >= 0:
        try:
            if words[index - 1].lower().strip() in pronouns:
                return True
        except:
            print(index)
            print(type(index))
            print(index - 1)
            print(len(words))
    if index + 1 < len(words):
        if words[index + 1].lower().strip() in pronouns:
            return True

    # if index + 2 < len(words):
    #     if words[index + 2].lower().strip() in pronouns:
    #         return True
    #
    # if index + 3 < len(words):
    #     if words[index + 3].lower().strip() in pronouns:
    #         return True

    return False

def instantiate_pos_dict(train_path, train_parse_path):
    # Read in the training files
    with open(train_path, 'r') as f:                
        train_tweets = []               #creates a list of the data per line
        for line in f:
            if line != '':
                train_tweets.append(line.split('\t'))

    with open(train_parse_path, 'r') as f:
        train_parse = []
        for line in f:
            if line != '':
                train_parse.append(line.split())

    assert len(train_tweets) == len(train_parse)

    # Create a dictionary to store the mapping for the one-hot
    counter = 1
    pos_dict = {"UNK": 0}

    # Look at the training files
    for j in range(1, len(train_tweets)):
        i = train_tweets[j]
        try:
            idx = int(i[0].split('_')[-1])              #idx is the intention function value of the vulgar word from the ID 
        except:
            print("j:" + str(j))
            print("i:" + str(i))
            print("ID: " + i[0])

        sent_len = len(i[4].split())                    #length of the tweet 

        parsed_words = train_parse[j]
        target_pos = parsed_words[idx].split('_')[-1]   # list of the POS for each word in the tweet 
        if target_pos not in pos_dict:                  # adds new POS to the POS dict 
            pos_dict[target_pos] = counter          # creates a unique value for each POS 
            counter += 1

        if idx != 0:                                                #finds the POS for the word preceding the vulgar word 
            before_pos = parsed_words[idx -1].split('_')[-1]
            if before_pos not in pos_dict:                          #adds the POS if the POS is not already in the dictionary 
                pos_dict[before_pos] = counter
                counter += 1
        else:
            if "START" not in pos_dict:                             # creates a pseudo start tag if the vulgar word appears at the beginning
                pos_dict["START"] = counter
                counter += 1

        if idx != (sent_len - 1):
            after_pos = parsed_words[idx + 1].split('_')[-1]        # finds the POS of the word following the vulgar word 
            if after_pos not in pos_dict:
                pos_dict[after_pos] = counter
                counter += 1
        else:   
            if "END" not in pos_dict:                               # creates a pseudo POS tag if the vulgar word is at the end of the tweet 
                pos_dict["END"] = counter
                counter += 1

    return pos_dict                                     # returns a dictionary with a POS and a unique value for each POS 


def one_hot_pos(pos, pos_dict):
    one_hot = np.zeros(len(pos_dict))
    one_hot[pos_dict[pos]] += 1

    return one_hot

def instantiate_target_dict(train_path):
    # Read in the training data
    with open(train_path, 'r') as f:
        train = f.read().split('\n')
    while '' in train:
        train.remove('')

    targets_dict = {}
    idx_counter = 0
    most_common = {}

    for i in train[1:]:
        line = i.split('\t')                    #creates a list of the elements in the line by: ID, URL, Tweet etc. 
        target = line[5].lower().strip()        #line[5] is the target word 
        if target not in targets_dict:          
            targets_dict[target] = idx_counter  # if the target word isn't in the dictionary then that words gets a value. each vulgar word gets a different value
            idx_counter += 1                    # one more than the previous word that was added to the dictionary 
        if target not in most_common:                   # if the word is not in the new most common dictionary 
            most_common[target] = [0, 0, 0, 0, 0, 0]    # then the value of the word gets an array of 1x5 
        most_common[target][int(line[12])] += 1         # if it is in the most common dictionary then the value increases at the consolidated value

    return targets_dict, most_common          # returns a dictionary of words and their unique value, and a dictionary of the most common function types for each word 

def one_hot_target(target, targets_dict):
    one_hot = np.zeros(len(targets_dict) + 1, dtype=int)        # creates an array from the size of the dictionary 
    if target in targets_dict:
        one_hot[targets_dict[target]] += 1                      #the array gets 1 added to it so it would be the only non-zero value given the vulgar word 
                                                                # target_dict[shit] = 0, so the one_hot for shit would have [1,0,0, ..., len(target_dict)+1] 

    else:
        one_hot[-1] += 1                                        #if the target word isn't in the dictionary then you created a psuedo value for the word 

    return one_hot

def one_hot_most_common(target, most_common_dict):
    one_hot = np.zeros(6, dtype=int)
    if target in most_common_dict:
        one_hot[np.argmax(most_common_dict[target])] += 1           #creates an array of the vulgar word's most common function and sets that as a one hot 
    else:
        one_hot[1] += 1

    return one_hot

def get_distribution_vec(target, most_common_dict):
    if target in most_common_dict:
        vec = most_common_dict[target]                          # is an array of size 6 and each element corresponds to a different function 
        num_uses = sum(vec)                                     # sums the values in the array 
        returnme = []                                           # creates a new list 
        for i in vec:                                           # creates a ratio for each element in the array 1/6 is the default 
            returnme.append(float(i)/num_uses)
        return returnme
    else:
        return [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]

def get_dist_sent(line_array):
    vec = np.zeros(6)
    i = 14
    j = 0
    while i< 20 :
        vec[j] = line_array[i]

        i += 1
        j += 1
    num_uses = sum(vec)
    returnme = []
    for i in vec:
        if num_uses == 0:
            returnme = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
        else:
            returnme.append(float(i)/num_uses)

    return returnme

def one_hot_sent(line_array):
    vec = np.zeros(6)
    i = int(line_array[20])
    vec[i] = 1
    '''if i == 0:
        vec[i] = 1 #what happens when the majority is non existent ? 
    else:
        vec[i] = 1'''
    return vec 


def clean_input_intention(train_path, train_parse_path, test_path, test_parse_path):
    print("Instantiating the Brown Cluster dictionary.")
    clusters = instantiate_clusters()
    print("Done.")

    print("Instantiating the TARGET dictionary for one-hot construction.")
    target_dict, most_common = instantiate_target_dict(train_path)
    print("Done.")

    print("Instantiating the POS dictionary for one-hot construction.")
    pos_dict = instantiate_pos_dict(train_path, train_parse_path)
    print("Done.")

    print("Reading in the embeddings.")
    embeddings = {}
    with open("/Users/victormena/Development/twitter/vulgartwitter/glove.twitter.27B/glove.twitter.27B.200d.txt", 'r') as f:
        for line in f:
            if line != '':
                line_list = line.split()
                embeddings[line_list[0]] = np.asarray(line_list[1:], dtype=float)
    print("Done.")

    #/Users/victormena/nltk_data/corpora/opinion_lexicon
    print("Reading in the positive sentiment lexicon.")
    positive_dict = set([])
    with open('/Users/victormena/nltk_data/corpora/opinion_lexicon/positive-words.txt', 'r') as f:
        for line in f:
            if line != '':
                if line[0] != ';':
                    positive_dict.add(line.strip())
    print("Done.")

    print("Reading in the negative sentiment lexicon.")
    negative_dict = set([])
    with open('/Users/victormena/nltk_data/corpora/opinion_lexicon/negative-words.txt', 'r') as f:

        for line in f:
            if line != '':
                if line[0] != ';':
                    negative_dict.add(line.strip())
    print("Done.")

    # Read in the training input
    with open(train_path, 'r') as f:
        train_tweets = f.read().split("\n")
    while '' in train_tweets:
        train_tweets.remove("")
    with open(train_parse_path, 'r') as f:
        train_parse = f.read().split('\n')
    while '' in train_parse:
        train_parse.remove('')

    
    # Read in the testing input
    with open(test_path, 'r') as f:
        test_tweets = f.read().split("\n")
    while '' in test_tweets:
        test_tweets.remove('')
    with open(test_parse_path, 'r') as f:
        test_parse = f.read().split('\n')
    while '' in test_parse:
        test_parse.remove('')

    train_cleaned = {"target": [], "label": [], "most_common": [], "distribution": [],  
                     "Majority": [], "sent_dist": [],                                               #creates dictionary for the different features 
                     "prev_pos": [], "target_pos": [], "next_pos": [], "context": [],
                     "positive": [], "negative": [], "pronoun": [], "brown_context": [],
                     "brown_whole": []}
    test_cleaned = {"target": [], "label": [], "most_common": [], "distribution": [],
                    "Majority": [], "sent_dist": [], 
                    "prev_pos": [], "target_pos": [], "next_pos": [], "context": [],
                    "positive": [], "negative": [], "pronoun": [], "brown_context": [],
                    "brown_whole": []}

    # Clean training set
    # add the sentiment and distribution to the training  
    # 3-9 for distribution
    # get what the function value is 12 
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
        # Extract the right information
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
        # create a function that gets the distribution of the tweet for sentiment 
        vec = get_dist_sent(line_array)
        train_cleaned['sent_dist'].append(vec)

        train_cleaned['target'].append(one_hot_target(line_array[5].lower().strip(), target_dict))              #creates a one hot array for each target word for POS 
        train_cleaned['label'].append(int(line_array[12]))                                                      # gives the function value for vulgar word in tweet 
        train_cleaned["most_common"].append(one_hot_most_common(line_array[5].lower().strip(),
                                                                most_common))
        train_cleaned['distribution'].append(get_distribution_vec(line_array[5].lower().strip(),                #gives distribution of function values 
                                                            most_common))
        train_cleaned['Majority'].append(one_hot_sent(line_array))
        # Extract the parts of speech
        pos_line = train_parse[line].split()                     # creates a list for each line(tweet)
        this_id = line_array[0]                                  # gives ID to each tweet 
        idx = int(this_id.split('_')[-1])                        # idx is the position for the vulgar word of the tweet 
        sent_len = len(pos_line)

        train_cleaned['target_pos'].append(one_hot_pos(pos_line[idx].split('_')[-1], pos_dict))         # reveals the POS using one hot       
        if idx != 0:
            train_cleaned['prev_pos'].append(one_hot_pos(pos_line[idx - 1].split('_')[-1], pos_dict))   #do the same thing for the for the previous word and the next word 
        else:
            train_cleaned['prev_pos'].append(one_hot_pos("START", pos_dict))
        if idx != (sent_len - 1):
            train_cleaned['next_pos'].append(one_hot_pos(pos_line[idx + 1].split('_')[-1], pos_dict))
        else:
            train_cleaned['next_pos'].append(one_hot_pos("END", pos_dict))

        # Get the context representation and sentiment features.
        text = line_array[4].lower()                                # gives the tweet 
        words = text.split()                                        # splits tweet word by word 

        sum_embeds = np.zeros(200)                                  # creates an array of 200 
        embedding_denom = 0
        positive_count = 0
        negative_count = 0
        for word in words:
            if word in positive_dict:
                positive_count += 1
            elif word.strip(string.punctuation) in positive_dict:
                positive_count += 1
            if word in negative_dict:
                negative_count += 1
            elif word.strip(string.punctuation) in negative_dict:
                negative_count += 1

            if word in embeddings:
                sum_embeds += embeddings[word]
                embedding_denom += 1
            elif word.strip(string.punctuation) in embeddings:
                sum_embeds += embeddings[word.strip(string.punctuation)]
                embedding_denom += 1
            else:
                pass

        if embedding_denom != 0:
            train_cleaned['context'].append(sum_embeds/embedding_denom)
        else:
            train_cleaned['context'].append(np.zeros(200))
        train_cleaned['positive'].append([positive_count / sent_len])
        train_cleaned['negative'].append([negative_count / sent_len])

        # Get the pronoun feature
        if check_pronoun(text, idx):
            train_cleaned['pronoun'].append([1])
        else:
            train_cleaned['pronoun'].append([0])

        # Get the RAW brown clusters feature (context)
        train_cleaned['brown_context'].append(get_brown_feature_context(text, idx, clusters))

        # Get the brown clusters feature (ALL)
        train_cleaned['brown_whole'].append(get_brown_feature_whole(text, clusters))
    #print(len(train_cleaned['Majority']))

    # Clean the test set
    df = pd.read_csv('cleaned_data_test.tsv', sep="\t")
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
    for line in range(1, len(test_tweets)):
        # Extract the right information
        line_array = test_tweets[line].split('\t')
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
        # create a function that gets the distribution of the tweet for sentiment 
        
        vec = get_dist_sent(line_array)
        test_cleaned['sent_dist'].append(vec)

        test_cleaned['target'].append(one_hot_target(line_array[5].lower().strip(), target_dict))
        test_cleaned['label'].append(int(line_array[12]))
        test_cleaned["most_common"].append(
            one_hot_most_common(line_array[5].lower().strip(), most_common))
        test_cleaned['distribution'].append(get_distribution_vec(line_array[5].lower().strip(), most_common))
        test_cleaned['Majority'].append(one_hot_sent(line_array))
        # Extract the parts of speech
        pos_line = test_parse[line].split()
        this_id = line_array[0]
        idx = int(this_id.split('_')[-1])
        sent_len = len(pos_line)

        targ = pos_line[idx].split('_')[-1]
        if targ in pos_dict:
            test_cleaned['target_pos'].append(one_hot_pos(targ, pos_dict))
        else:
            test_cleaned['target_pos'].append(one_hot_pos("UNK", pos_dict))

        if idx != 0:
            prev = pos_line[idx - 1].split('_')[-1]
            if prev in pos_dict:
                test_cleaned['prev_pos'].append(one_hot_pos(prev, pos_dict))
            else:
                test_cleaned['prev_pos'].append(one_hot_pos("UNK", pos_dict))
        else:
            test_cleaned['prev_pos'].append(one_hot_pos("START", pos_dict))

        if idx != (sent_len - 1):
            next = pos_line[idx + 1].split('_')[-1]
            if next in pos_dict:
                test_cleaned['next_pos'].append(one_hot_pos(next, pos_dict))
            else:
                test_cleaned['next_pos'].append(one_hot_pos("UNK", pos_dict))
        else:
            test_cleaned['next_pos'].append(one_hot_pos("END", pos_dict))

        # Get the context representation.
        text = line_array[4].lower()
        words = text.split()

        sum_embeds = np.zeros(200)
        embedding_denom = 0
        positive_count = 0
        negative_count = 0

        for word in words:
            if word in positive_dict:
                positive_count += 1
            elif word.strip(string.punctuation) in positive_dict:
                positive_count += 1
            if word in negative_dict:
                negative_count += 1
            elif word.strip(string.punctuation) in negative_dict:
                negative_count += 1

            if word in embeddings:
                sum_embeds += embeddings[word]
                embedding_denom += 1
            elif word.strip(string.punctuation) in embeddings:
                sum_embeds += embeddings[word.strip(string.punctuation)]
                embedding_denom += 1
            else:
                pass
        if embedding_denom != 0:
            test_cleaned['context'].append(sum_embeds / embedding_denom)
        else:
            test_cleaned['context'].append(np.zeros(200))
        test_cleaned['positive'].append([positive_count / sent_len])
        test_cleaned['negative'].append([negative_count / sent_len])

        # Get the pronoun feature
        if check_pronoun(text, idx):
            test_cleaned['pronoun'].append([1])
        else:
            test_cleaned['pronoun'].append([0])

        # Get the RAW brown clusters feature (context)
        test_cleaned['brown_context'].append(get_brown_feature_context(text, idx, clusters))

        # Get the RAW brown clusters feature (ALL)
        test_cleaned['brown_whole'].append(get_brown_feature_whole(text, clusters))
    
    return train_cleaned, test_cleaned, target_dict, most_common




'''def leave_one_out(clean_train, clean_test):
    # need to go through each tweet from both clean_train and clean_test 
    new_x = {"target": [], "label": [], "most_common": [], "distribution": [],
                    "Majority": [], "sent_dist": [], 
                    "prev_pos": [], "target_pos": [], "next_pos": [], "context": [],
                    "positive": [], "negative": [], "pronoun": [], "brown_context": [],
                    "brown_whole": []}


    for i in clean_train:
        merge = clean_train[i] + clean_test[i]
        new_x[i] = merge 

    return new_x'''



class LogRegModel:
    def __init__(self, params=None):
        if params is not None:
            self.params = params
        else:
            self.params = {
                "solver": "sag",
                "multi_class": "ovr",
            }
        #best max_iter was 1050...
        self.model = sklearn.linear_model.LogisticRegression(solver=self.params['solver'],
                                                             multi_class=self.params['multi_class'],
                                                             #class_weight = 'balanced',
                                                             max_iter=1000)

    def init_train(self, cleaned):
        # Pull the target word feature (one-hot)
        all_targets = []
        for i in cleaned['target']:
            all_targets.append(i)

        # Pull the gold labels
        all_y = []
        for i in cleaned['label']:
            all_y.append(int(i))

        # Pull the most common feature (one-hot)
        all_mc = []
        for i in cleaned['most_common']:
            all_mc.append(i)

        # Pull the distribution feature group
        all_dist = []
        for i in cleaned['distribution']:
            all_dist.append(i)

        # Pull all the majority feature group
        all_maj = []
        for i in cleaned['Majority']:
            all_maj.append(i)

        # Pull all the sentiment distributions feature group
        all_sent = []
        for i in cleaned['sent_dist']:
            all_sent.append(i)

        # Pull the POS feature group
        all_prev_pos = []
        for i in cleaned['prev_pos']:
            all_prev_pos.append(i)

        all_target_pos = []
        for i in cleaned['target_pos']:
            all_target_pos.append(i)
        all_next_pos = []
        for i in cleaned['next_pos']:
            all_next_pos.append(i)

        # Pull the context representation
        all_context = []
        for i in cleaned['context']:
            all_context.append(i)

        # Pull the sentiment information
        all_positive = []
        for i in cleaned['positive']:
            all_positive.append(i)
            # all_positive.append(i[0])
        self.positive_max = max(all_positive)
        self.positive_min = min(all_positive)

        # all_positive_minmax = []
        # for i in all_positive:
        #     all_positive_minmax.append([(i - self.positive_min)/(self.positive_max - self.positive_min)])

        all_negative = []
        for i in cleaned['negative']:
            all_negative.append(i)
            # all_negative.append(i[0])
        self.negative_max = max(all_negative)
        self.negative_min = min(all_negative)

        # all_negative_minmax = []
        # for i in all_negative:
        #     all_negative_minmax.append([(i - self.negative_min)/(self.negative_max - self.negative_min)])

        all_positive_indicator = []
        for i in range(len(all_positive)):
            pos_count = all_positive[i][0]
            neg_count = all_negative[i][0]

            if pos_count > neg_count:
                all_positive_indicator.append([1])
            else:
                all_positive_indicator.append([0])

        all_pronoun = []
        for i in cleaned['pronoun']:
            all_pronoun.append(i)

        # Pull the brown features and turn them into one-hots
        all_brown_raw = []
        for i in cleaned['brown_context']:
            all_brown_raw.append(i)

        brown_one_hot_mapping = cluster_one_hot_dict(all_brown_raw)
        all_p_one_hot = []
        all_n_one_hot = []
        for i in all_brown_raw:
            p_one_hot, n_one_hot = one_hot_cluster(i, brown_one_hot_mapping)
            all_p_one_hot.append(p_one_hot)
            all_n_one_hot.append(n_one_hot)

        # # Pull the brown ALL features and turn them into one hots
        all_brown_raw_whole = []
        for i in cleaned['brown_whole']:
             all_brown_raw_whole.append(i)
        brown_one_hot_mapping_whole = cluster_one_hot_dict_whole(all_brown_raw_whole)
        all_brown_whole = []
        for i in all_brown_raw_whole:
             all_brown_whole.append(one_hot_cluster_whole(i, brown_one_hot_mapping_whole))


        self.train_targets = all_targets
        self.train_y = all_y
        self.train_most_common = all_mc
        self.train_distribution = all_dist
        self.train_majority = all_maj
        self.train_sent_dist = all_sent 
        self.train_prev_pos = all_prev_pos
        self.train_target_pos = all_target_pos
        self.train_next_pos = all_next_pos
        self.train_context = all_context
        self.train_positive = all_positive
        # self.train_positive_minmax = all_positive_minmax
        self.train_negative = all_negative
        # self.train_negative_minmax = all_negative_minmax
        self.train_positive_indicator = all_positive_indicator
        self.train_pronoun = all_pronoun
        self.train_prev_brown = all_p_one_hot
        self.train_next_brown = all_n_one_hot
        # self.train_all_brown = all_brown_whole

        # print("Train_targets example:" + str(all_targets[0]))
        # print("Train_y example: " + str(all_y[0]))
        # print("Train_Most_Common example:" + str(all_mc[0]))
        # print("Train_Distribution example:" + str(all_dist[0]))
        # print(len(all_dist))

        return brown_one_hot_mapping , brown_one_hot_mapping_whole

    def init_test(self, cleaned, brown_one_hot_mapping, brown_one_hot_mapping_whole):
        # Pull the target word feature (one-hot)
        all_targets = []
        for i in cleaned['target']:
            all_targets.append(i)

        # Pull the gold labels
        all_y = []
        for i in cleaned['label']:
            all_y.append(int(i))
        # Pull the most common feature (one-hot)
        all_mc = []
        for i in cleaned['most_common']:
            all_mc.append(i)

        # Pull the distribution feature group
        all_dist = []
        for i in cleaned['distribution']:
            all_dist.append(i)

        # Pull all the majority feature group
        all_maj = []
        for i in cleaned['Majority']:
            all_maj.append(i)

        # Pull all the sentiment distributions feature group
        all_sent = []
        for i in cleaned['sent_dist']:
            all_sent.append(i)


        # Pull the POS feature group
        all_prev_pos = []
        for i in cleaned['prev_pos']:
            all_prev_pos.append(i)
        all_target_pos = []
        for i in cleaned['target_pos']:
            all_target_pos.append(i)
        all_next_pos = []
        for i in cleaned['next_pos']:
            all_next_pos.append(i)

        # Pull the context feature
        all_context = []
        for i in cleaned['context']:
            # all_context.append(i)
            all_context.append(i)

        # Pull the sentiment information
        all_positive = []
        for i in cleaned['positive']:
            all_positive.append(i)
            # all_positive.append(i[0])

        # all_positive_minmax = []
        # for i in all_positive:
        #     adjusted = (i - self.positive_min) / (self.positive_max - self.positive_min)
        #     if adjusted <= 1:
        #         all_positive_minmax.append([adjusted])
        #     else:
        #         all_positive_minmax.append([0])

        all_negative = []
        for i in cleaned['negative']:
             all_negative.append(i)
            # all_negative.append(i[0])
        # all_negative_minmax = []
        # for i in all_negative:
        #     adjusted = (i - self.negative_min) / (self.negative_max - self.negative_min)
        #     if adjusted <= 1:
        #         all_negative_minmax.append([adjusted])
        #     else:
        #         all_negative_minmax.append([0])

        all_positive_indicator = []
        for i in range(len(all_positive)):
            pos_count = all_positive[i][0]
            neg_count = all_negative[i][0]

            if pos_count > neg_count:
                all_positive_indicator.append([1])
            else:
                all_positive_indicator.append([0])

        all_pronoun = []
        for i in cleaned['pronoun']:
            all_pronoun.append(i)

        # Get the brown feature
        brown_raw = []
        for i in cleaned['brown_context']:
            brown_raw.append(i)
        all_p_one_hot = []
        all_n_one_hot = []
        for i in brown_raw:
            p_one_hot, n_one_hot = one_hot_cluster(i, brown_one_hot_mapping)
            all_p_one_hot.append(p_one_hot)
            all_n_one_hot.append(n_one_hot)

        # # Pull the brown ALL features and turn them into one hots
        # # Pull the brown ALL features and turn them into one hots
        all_brown_raw_whole = []
        for i in cleaned['brown_whole']:
             all_brown_raw_whole.append(i)
        #
        all_brown_whole = []
        for i in all_brown_raw_whole:
             all_brown_whole.append(one_hot_cluster_whole(i, brown_one_hot_mapping_whole))

        self.test_targets = all_targets
        self.test_y = all_y
        self.test_most_common = all_mc
        self.test_distribution = all_dist
        self.test_majority = all_maj
        self.test_sent_dist = all_sent 
        self.test_prev_pos = all_prev_pos
        self.test_target_pos = all_target_pos
        self.test_next_pos = all_next_pos
        self.test_context = all_context
        self.test_positive = all_positive
        self.test_negative = all_negative
        # self.test_positive_minmax = all_positive_minmax
        # self.test_negative_minmax = all_negative_minmax
        self.test_positive_indicator = all_positive_indicator
        self.test_pronoun = all_pronoun
        self.test_prev_brown = all_p_one_hot
        self.test_next_brown = all_n_one_hot
        self.test_all_brown = all_brown_whole

        # print("Test_targets example:" + str(all_targets[0]))
        # print("Test_y example: " + str(all_y[0]))
        # print("Test_Most_Common example:" + str(all_mc[0]))
        # print("Test_Distribution example:" + str(all_dist[0]))
        # print(len(all_dist))



    def concat(self):
        new_distribution = self.train_distribution + self.test_distribution
        new_sent_dist = self.train_sent_dist + self.test_sent_dist
        new_majority = self.train_majority + self.test_majority
        new_prev = self.train_prev_pos + self.test_prev_pos
        new_target = self.train_target_pos + self.test_target_pos
        new_next = self.train_next_pos + self.test_next_pos
        new_context = self.train_context + self.test_context
        new_positive = self.train_positive + self.test_positive
        new_negative = self.train_negative + self.test_negative
        new_prev_brown = self.train_prev_brown + self.test_prev_brown
        new_next_brown = self.train_next_brown + self.test_next_brown
        new_y = self.train_y + self.test_y 



        feats = [new_distribution, new_sent_dist, new_majority, new_prev, new_target, new_next, 
                new_context, new_positive, new_negative, new_prev_brown, new_next_brown]
        y = new_y

        dims = len(feats[0][0])
        train_feats = feats[0]

        if len(feats) > 1:
            for i in feats[1:]:
                train_feats = np.append(train_feats, i, axis =1 )
                dims += len(i[0])

        return train_feats, y 

    def loo_train(self, train, y):
        self.model.fit(train, y)

    def loo_test(self, test, y):
        y = [y]
        predictions = self.model.predict(test)
        print("Len Predictions:" + str(len(predictions)))
        print("Len Test_Y: " + str(len(y)))

        mic_f1 = f1_score(y, predictions, labels=None, pos_label=1,
                          average='micro', sample_weight=None)
        mac_f1 = f1_score(y, predictions, labels=None, pos_label=1,
                          average='macro', sample_weight=None)
        acc = accuracy_score(y, predictions, normalize=True, sample_weight=None)
        prec = precision_score(y, predictions, labels=None, pos_label=1,
                               average='macro')
        rec = recall_score(y, predictions, labels=None, pos_label=1, average='macro')
        return mic_f1, mac_f1, acc, prec, rec 





    def train(self):
        feats = [self.train_distribution, 
                    self.train_sent_dist, self.train_majority,
                     self.train_prev_pos, self.train_target_pos,
                    self.train_next_pos, self.train_context, self.train_positive, self.train_negative,
                 self.train_prev_brown, self.train_next_brown]

        dims = len(feats[0][0])         # length of train_distribution array 
        train_feats = feats[0]          #the train_distribution array 
        
        if len(feats) > 1:              # if len(feats) is more than 1 which it is idk why 
            # a loop to go through the feat list starting from 1 to the end. (train_sent_dist to train_next_brown)
            for i in feats[1:]:
                # appends the train_feats with the next array until it reaches the last element of the feats array                                      
                train_feats = np.append(train_feats, i, axis=1) 

                dims += len(i[0])
        #length of train_feats is the number of tweets in the training data 
        print("Printing Training Feature Dimensions:")
        for i in feats:
            print(str(len(i[0])))
        print("Train Feats Dims: " + str(len(train_feats[0])))
    
        self.model.fit(train_feats, self.train_y)


    def test(self):
        feats = [self.test_distribution, 
                  self.test_sent_dist, self.test_majority,
                  self.test_prev_pos, self.test_target_pos,
                 self.test_next_pos, self.test_context, self.test_positive, self.test_negative,
                 self.test_prev_brown, self.test_next_brown]
        # feats = [self.test_prev_pos, self.test_target_pos,
        #          self.test_next_pos, self.test_context, self.test_positive, self.test_negative]
        dims = len(feats[0][0])
        test_feats = feats[0]
        if len(feats) > 1:
            for i in feats[1:]:
                test_feats = np.append(test_feats, i, axis=1)
                dims += len(i[0])

        print("Printing Testing Feature Dimensions:")
        for i in feats:
            print(str(len(i[0])))
        print("Test Feats Dims: " + str(len(test_feats[0])))

        predictions = self.model.predict(test_feats)
        print("Len Predictions:" + str(len(predictions)))
        print("Len Test_Y: " + str(len(self.test_y)))

        mic_f1 = f1_score(self.test_y, predictions, labels=None, pos_label=1,
                          average='micro', sample_weight=None)
        mac_f1 = f1_score(self.test_y, predictions, labels=None, pos_label=1,
                          average='macro', sample_weight=None)
        acc = accuracy_score(self.test_y, predictions, normalize=True, sample_weight=None)
        prec = precision_score(self.test_y, predictions, labels=None, pos_label=1,
                               average='macro')
        rec = recall_score(self.test_y, predictions, labels=None, pos_label=1, average='macro')

        print("Micro: " + str(mic_f1))
        print("Macro: " + str(mac_f1))
        print("Acc: " + str(acc))
        print("Precision: " + str(prec))
        print("Recall: " + str(rec))






def main():
    logreg = LogRegModel()
    # print("Cleaning the data.")
    clean_train, clean_test, target_dict, most_common = clean_input_intention("train_set.txt",
                                                                               "train_tagged.txt",
                                                                              "test_set.txt",
                                                                             "test_tagged.txt")
    #
    brown_one_hot_mapping, brown_one_hot_mapping_whole = logreg.init_train(clean_train)
    logreg.init_test(clean_test, brown_one_hot_mapping, brown_one_hot_mapping_whole)
    # concatenate the training and testing data 
    feats, y  = logreg.concat()
    #go through a loop to do the training and testing of the model 
    micro = 0
    macro = 0 
    acc = 0 
    precision = 0 
    recall = 0 
    for i in range(10):
        # the first element in the feats, y 
        feats, y  = logreg.concat()
        test = feats[i].reshape(1,-1)
        test_y = y[i]
        train = np.delete(feats, i, axis = 0)
        y.pop(i)
        train_y = y
        logreg.loo_train(train, train_y)
        micro_1, macro_1, acc_1, precision_1, recall_1 = logreg.loo_test(test, test_y)
        micro += micro_1
        macro += macro_1 
        acc += acc_1
        precision += precision_1 
        recall += recall_1
    print(micro/10) 











main()