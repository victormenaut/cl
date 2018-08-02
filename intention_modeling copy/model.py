from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, concatenate
from keras.engine import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import optimizers
from sklearn.metrics import f1_score, accuracy_score

import logging
import numpy as np
import argparse

def instantiate_pos_dict(train_path, train_parse_path, test_path, test_parse_path, val_path, val_parse_path):
    # Read in the training files
    with open(train_path, 'r') as f:
        train_tweets = []
        for line in f:
            if line != '':
                train_tweets.append(line.split('\t'))

    with open(train_parse_path, 'r') as f:
        train_parse = []
        for line in f:
            if line != '':
                train_parse.append(line.split())

    # with open(train_path, 'r') as f:
    #     train_tweets = f.read().split('\n')
    # while '' in train_tweets:
    #     train_tweets.remove('')
    #
    # with open(train_parse_path, 'r') as f:
    #     train_parse = f.read().split('\n')
    # while '' in train_parse:
    #     train_parse.remove('')


    assert len(train_tweets) == len(train_parse)

    # Read in the testing files
    with open(test_path, 'r') as f:
        test_tweets = []
        for line in f:
            if line != '':
                test_tweets.append(line.split('\t'))

    with open(test_parse_path, 'r') as f:
        test_parse = []
        for line in f:
            if line != '':
                test_parse.append(line.split())

    assert len(test_tweets) == len(test_parse)

    # Read in the validation files
    with open(val_path, 'r') as f:
        val_tweets = []
        for line in f:
            if line != '':
                val_tweets.append(line.split('\t'))

    with open(val_parse_path, 'r') as f:
        val_parse = []
        for line in f:
            if line != '':
                val_parse.append(line.split())

    assert len(val_tweets) == len(val_parse)

    # Create a dictionary to store the mapping for the one-hot
    counter = 0
    pos_dict = {}

    # Look at the training files
    for j in range(1, len(train_tweets)):
        i = train_tweets[j]                     #the line of the training set 
        try:
            idx = int(i[0].split('_')[-1])      #lets you know the position of the cuss word 
        except:
            print("j:" + str(j))
            print("i:" + str(i))
            print("ID: " + i[0])

        sent_len = len(i[4].split())            #this is the actual tweet length 

        parsed_words = train_parse[j]           #this is the list of the pos for each word in a tweet 
        target_pos = parsed_words[idx].split('_')[-1]   #this tells you the part of speech of the vulgar word. idx is the position of the vulgar word 
        if target_pos not in pos_dict:
            pos_dict[target_pos] = counter
            counter += 1

        if idx != 0:
            before_pos = parsed_words[idx -1].split('_')[-1]       # finds the previous words POS
            if before_pos not in pos_dict:
                pos_dict[before_pos] = counter
                counter += 1
        else:
            if "START" not in pos_dict:         #the vulgar word is at the beginning of the tweet not a vulgar word 
                pos_dict["START"] = counter
                counter += 1

        if idx != (sent_len - 1):                                   
            after_pos = parsed_words[idx + 1].split('_')[-1]        #finds the word after vulgar POS 
            if after_pos not in pos_dict:
                pos_dict[after_pos] = counter
                counter += 1
        else:
            if "END" not in pos_dict:               #the vulgar is at the end of the tweet, not a vulgar word 
                pos_dict["END"] = counter
                counter += 1

    # Look at the test files
    for j in range(1, len(test_tweets)):
        i = test_tweets[j]
        idx = int(i[0].split('_')[-1])
        sent_len = len(i[4].split())

        parsed_words = test_parse[j]
        try:
            target_pos = parsed_words[idx].split('_')[-1]
        except:
            print(str(parsed_words))
            print(str(len(parsed_words)))
            print(str(idx))
            print(str(parsed_words[idx]))
        if target_pos not in pos_dict:
            pos_dict[target_pos] = counter
            counter += 1

        if idx != 0:
            before_pos = parsed_words[idx -1].split('_')[-1]
            if before_pos not in pos_dict:
                pos_dict[before_pos] = counter
                counter += 1
        else:
            if "START" not in pos_dict:
                pos_dict["START"] = counter
                counter += 1

        if idx != (sent_len - 1):
            after_pos = parsed_words[idx + 1].split('_')[-1]
            if after_pos not in pos_dict:
                pos_dict[after_pos] = counter
                counter += 1
        else:
            if "END" not in pos_dict:
                pos_dict["END"] = counter
                counter += 1

    # Look at the validation files
    for j in range(1, len(val_tweets)):
        i = val_tweets[j]
        idx = int(i[0].split('_')[-1])
        sent_len = len(i[4].split())

        parsed_words = val_parse[j]
        target_pos = parsed_words[idx].split('_')[-1]
        if target_pos not in pos_dict:
            pos_dict[target_pos] = counter
            counter += 1

        if idx != 0:
            before_pos = parsed_words[idx -1].split('_')[-1]
            if before_pos not in pos_dict:
                pos_dict[before_pos] = counter
                counter += 1
        else:
            if "START" not in pos_dict:
                pos_dict["START"] = counter
                counter += 1

        if idx != (sent_len - 1):
            after_pos = parsed_words[idx + 1].split('_')[-1]
            if after_pos not in pos_dict:
                pos_dict[after_pos] = counter
                counter += 1
        else:
            if "END" not in pos_dict:
                pos_dict["END"] = counter
                counter += 1

    return pos_dict, len(pos_dict)


def one_hot_pos(pos, pos_dict):
    one_hot = np.zeros(len(pos_dict))
    one_hot[pos_dict[pos]] += 1

    return one_hot

def instantiate_target_dict(train_path, test_path, val_path):
    # Read in the training data
    with open(train_path, 'r') as f:
        train = f.read().split('\n')            #train is a list of each line per training data 
    while '' in train:                          #removes any empty lines 
        train.remove('')

    # Read in the testing data
    with open(test_path, 'r') as f:
        test = f.read().split('\n')
    while '' in test:
        test.remove('')

    # Read in the validation data
    with open(val_path, 'r') as f:
        val = f.read().split('\n')
    while '' in val:
        val.remove('')

    targets_dict = {}
    idx_counter = 0

    for i in train[1:]:
        line = i.split('\t')
        target = line[5].lower().strip()            #finds target word 
        if target not in targets_dict:              
            targets_dict[target] = idx_counter      # adds a count for each new word 
            idx_counter += 1

    for i in test[1:]:
        line = i.split('\t')
        target = line[5].lower().strip()
        if target not in targets_dict:
            targets_dict[target] = idx_counter
            idx_counter += 1

    for i in val[1:]:
        line = i.split('\t')
        target = line[5].lower().strip()
        if target not in targets_dict:
            targets_dict[target] = idx_counter
            idx_counter += 1

    return targets_dict

def one_hot_target(target, targets_dict):
    one_hot = np.zeros(len(targets_dict))
    one_hot[targets_dict[target]] += 1

    return one_hot


def clean_input_intention(train_path, train_parse_path, test_path, test_parse_path, val_path, val_parse_path): # embeddings_path):
    print("Instantiating the POS Dictionary for one-hot construction.")
    pos_dict, pos_dims = instantiate_pos_dict(train_path, train_parse_path, test_path, test_parse_path, val_path, val_parse_path)
    print("Done.")
    print("Instantiating the TARGET dictionary for one-hot construction.")
    target_dict = instantiate_target_dict(train_path, test_path, val_path)
    print("Done.")

    # Read in the training input
    with open(train_path, 'r') as f:
        train_tweets = f.read().split("\n")
    f.close()
    while '' in train_tweets:
        train_tweets.remove("")

    # Read in the training parse
    with open(train_parse_path, 'r') as f:
        train_parse = f.read().split('\n')
    while '' in train_parse:
        train_parse.remove('')

    # Read in the testing input
    with open(test_path, 'r') as f:
        test_tweets = f.read().split("\n")
    f.close()
    while '' in test_tweets:
        test_tweets.remove('')

    # Read in the testing parse
    with open(test_parse_path, 'r') as f:
        test_parse = f.read().split('\n')
    while '' in test_parse:
        test_parse.remove('')

    # Read in the validation input
    with open(val_path, 'r') as f:
        val_tweets = f.read().split('\n')
    while '' in val_tweets:
        val_tweets.remove('')

    # Read in the validation parse
    with open(val_parse_path, 'r') as f:
        val_parse = f.read().split('\n')
    while '' in val_parse:
        val_parse.remove('')

    # # Read in the embedding dictionary
    # embeds = []
    # embed_dim = None

    # with open(embeddings_path) as f:
    #     # Loop over the lines of the file. Each line contains a word
    #     # followed by a vector
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         coefs = np.asarray(values[1:], dtype='float32')
    #
    #         # If this is the first word, get the number of dimensions
    #         if embed_dim is None:
    #             embed_dim = len(coefs)
    #             print("embed_dim set to " + str(embed_dim))
    #
    #         # Normalize the embeddings to the unit sphere
    #         embeds[word] = coefs / np.linalg.norm(coefs)
    # f.close()

    train_cleaned = {"idx": [], "tweet": [], "label" : [], "prev_vec": [], "target_vec": [], "next_vec": [], "prev_POS": [], "target_POS": [], "next_POS": [], "target": []}
    test_cleaned = {"idx": [], "tweet": [], "label" : [], "prev_vec": [], "target_vec": [], "next_vec": [], "prev_POS": [], "target_POS": [], "next_POS": [], "target": []}
    val_cleaned = {"idx": [], "tweet": [], "label": [], "prev_vec": [], "target_vec": [],
                    "next_vec": [], "prev_POS": [], "target_POS": [], "next_POS": [], "target": []}
    # Tweet_In = namedtuple('Tweet_In', ['id', 'tweet', 'idx', 'label'])

    # Clean training set
    for line in range(1, len(train_tweets)):
        # Extract the right information
        line_array = train_tweets[line].split('\t')
        ids = line_array[0]
        tweet = line_array[4]
        label = line_array[12]
        idx = int(ids.split('_')[-1])

        # Get the list of tagged tokens for this tweet
        token_list = train_parse[line].split()              #splits each line into a list of words with their part of speech        

        # Determine which instance of the target this is (i.e., if the target
        # is 'shit', is it the first occurrence of shit? The second?
        # First, we determine what word the target actually is
        words = tweet.split()                               #splits each word in the tweet into a list 

        # Append all the information
        # Basic features
        train_cleaned['idx'].append(float(idx)/len(token_list)) #ratio of position of vulgar word to the length of the tweet 
        train_cleaned['tweet'].append(tweet)                   
        train_cleaned['label'].append(int(label))               #majority 
        train_cleaned['target'].append(one_hot_target(line_array[5].lower().strip(), target_dict))     #an array for the one hot position of the target word 

        ## POS features
        # Previous
        if idx != 0:
            train_cleaned['prev_POS'].append(one_hot_pos(token_list[idx - 1].split('_')[-1], pos_dict))         # finds one hot for prev word POS 
        else:
            train_cleaned['prev_POS'].append(one_hot_pos('START', pos_dict))

        # Target
        train_cleaned['target_POS'].append(one_hot_pos(token_list[idx].split('_')[-1], pos_dict))

        # Next
        if idx != len(words) - 1:
            train_cleaned['next_POS'].append(
                one_hot_pos(token_list[idx + 1].split('_')[-1], pos_dict))
        else:
            train_cleaned['next_POS'].append(one_hot_pos('END',pos_dict))

        # vec features
        train_cleaned['prev_vec'] = None
        train_cleaned['target_vec'] = None
        train_cleaned['next_vec'] = None

    for line in range(1, len(test_tweets)):
        # Extract the right information
        line_array = test_tweets[line].split('\t')
        ids = line_array[0]
        tweet = line_array[4]
        label = line_array[12]
        idx = int(ids.split('_')[-1])

        # Get the list of tagged tokens for this tweet
        token_list = test_parse[line].split()

        # Determine which instance of the target this is (i.e., if the target
        # is 'shit', is it the first occurrence of shit? The second?
        # First, we determine what word the target actually is
        words = tweet.split()

        # Append all the information
        # Basic features
        test_cleaned['idx'].append(float(idx)/len(token_list))
        test_cleaned['tweet'].append(tweet)
        test_cleaned['label'].append(int(label))
        test_cleaned['target'].append(one_hot_target(line_array[5].lower().strip(), target_dict))

        ## POS features
        # Previous
        if idx != 0:
            test_cleaned['prev_POS'].append(one_hot_pos(token_list[idx - 1].split('_')[-1], pos_dict))
        else:
            test_cleaned['prev_POS'].append(one_hot_pos('START', pos_dict))

        # Target
        test_cleaned['target_POS'].append(one_hot_pos(token_list[idx].split('_')[-1], pos_dict))

        # Next
        if idx != len(words) - 1:
            test_cleaned['next_POS'].append(
                one_hot_pos(token_list[idx + 1].split('_')[-1], pos_dict))
        else:
            test_cleaned['next_POS'].append(one_hot_pos('END',pos_dict))

        # vec features
        train_cleaned['prev_vec'] = None
        train_cleaned['target_vec'] = None
        train_cleaned['next_vec'] = None

    # Clean the validation set
    for line in range(1, len(val_tweets)):
        # Extract the right information
        line_array = val_tweets[line].split('\t')
        ids = line_array[0]
        tweet = line_array[4]
        label = line_array[12]
        idx = int(ids.split('_')[-1])

        # Get the list of tagged tokens for this tweet
        token_list = val_parse[line].split()

        # Determine which instance of the target this is (i.e., if the target
        # is 'shit', is it the first occurrence of shit? The second?
        # First, we determine what word the target actually is
        words = tweet.split()

        # Append all the information
        # Basic features
        val_cleaned['idx'].append(float(idx)/len(token_list))
        val_cleaned['tweet'].append(tweet)
        val_cleaned['label'].append(int(label))
        val_cleaned['target'].append(one_hot_target(line_array[5].lower().strip(), target_dict))

        ## POS features
        # Previous
        if idx != 0:
            val_cleaned['prev_POS'].append(one_hot_pos(token_list[idx - 1].split('_')[-1], pos_dict))
        else:
            val_cleaned['prev_POS'].append(one_hot_pos('START', pos_dict))

        # Target
        val_cleaned['target_POS'].append(one_hot_pos(token_list[idx].split('_')[-1], pos_dict))

        # Next
        if idx != len(words) - 1:
            val_cleaned['next_POS'].append(
                one_hot_pos(token_list[idx + 1].split('_')[-1], pos_dict))
        else:
            val_cleaned['next_POS'].append(one_hot_pos('END',pos_dict))

        # vec features
        val_cleaned['prev_vec'] = None
        val_cleaned['target_vec'] = None
        val_cleaned['next_vec'] = None

    return train_cleaned, test_cleaned, val_cleaned, pos_dims


class BiLSTM:

    def __init__(self, params):
        """
            This __init__ function can be called without an explicit argument in order to
            utilize the default parameters for the network.


            :param params: a dictionary of parameters, see the inside of the function for
                           examples
        """
        if params is not None:
            self.params = params
        else:
            self.params = {"sysid": "attn_len",
                           "embed_dim": None,
                           "train_embed": False,
                           "rnndim": 128,
                           "dropout": 0.2,
                           "validation_split": 0.05,
                           "nepoch": 10,
                           "batch_size": 64,
                           "maxsentlen": 60,
                           "lr":0.001
                           }

        # Set the initial embedding weights
        #self.params['initial_embed_weights'] = "/Users/eholgate/Desktop/cbow_200.txt"

    def load(self, filename):
        """
            This is a function to load a model from hdf5.
            :param filename: the path to the hdf5
        """
        self.model = load_model(filename)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def init_train(self, cleaned):
        """
            This is a function to initialize the training data.
            This function does not return anything. Instead, it sets instance
            fields to the appropriate values.

            :param allx: a list of input
            :param ally: a list of gold standard output
        """
        allx = []                                       # a list of all the tweets 
        ally = []                                       # labels for each tweet 
        all_idx = []                                    # list of all the IDs for the tweets 
        all_prev_pos = cleaned['prev_POS']              # list of all the POS for the previous word 
        all_target_pos = cleaned['target_POS']          # list of all the POS for the target word 
        all_next_pos = cleaned['next_POS']              # list of all the POS for the next word 
        all_targets = cleaned['target']                 # list of all the target words in each tweet 

        for tweet in cleaned['tweet']:
            allx.append(tweet)
        for label in cleaned['label']:
            one_hot = [0, 0, 0, 0, 0,0]
            one_hot[int(label)] += 1

            ally.append(one_hot)
        for idx in cleaned['idx']:                      
            all_idx.append([idx])

        if params['feat_groups'] == '':                 #trains just based off the ID
            self.train_feats = np.asarray(all_idx)
        elif params['feat_groups'] == 'pos_only':       #trains just based off the POS given ID 
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            self.train_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array), axis=1)
        elif params['feat_groups'] == 'all':                            #takes everything and puts it into one huge array 
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            all_targets_array = np.asarray(all_targets)
            self.train_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array, all_targets_array), axis=1)



        #self.train_idx = np.asarray(all_idx)

        # Set the max sentence length, which determines what to pad to
        self.maxlen = self.params["maxsentlen"]

        # Construct a Tokenizer to determine how to tokenize the input
        self.tokenizer = Tokenizer(split=" ")

        # Fit the tokenizer on the input and output the fitted sequences
        self.tokenizer.fit_on_texts(allx)
        sequences = self.tokenizer.texts_to_sequences(allx)
        logging.info('Found %s unique tokens.' % len(self.tokenizer.word_index))

        # Pad the sequences and save the training data
        self.xtrain_tweets = pad_sequences(sequences, maxlen=self.maxlen)

        self.ytrain = np.asarray(ally)


        # Log relevant information
        logging.info('Shape of X: {0}'.format(self.xtrain_tweets.shape))
        logging.info('Shape of Y: {0}'.format(self.ytrain.shape))
        logging.info("train data init complete")


    def init_test(self, cleaned):
        """
            This is a function to initialize the training data.
            This function does not return anything. Instead, it sets instance
            fields to the appropriate values.

            :param allx: a list of input
            :param ally: a list of gold standard output
        """
        assert (hasattr(self, "tokenizer")), "This is for testing; init training!"
        allx = []
        ally = []
        all_idx = []

        for tweet in cleaned['tweet']:
            allx.append(tweet)
        for label in cleaned['label']:
            ally.append(label)

        for idx in cleaned['idx']:
            all_idx.append([idx])

        all_prev_pos = cleaned['prev_POS']
        all_target_pos = cleaned['target_POS']
        all_next_pos = cleaned['next_POS']
        all_targets = cleaned['target']

        if params['feat_groups'] == '':
            self.test_feats = np.asarray(all_idx)
        elif params['feat_groups'] == 'pos_only':
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            self.test_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array), axis=1)
        elif params['feat_groups'] == 'all':
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            all_targets_array = np.asarray(all_targets)
            self.test_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array, all_targets_array), axis=1)

        self.tokenizer.fit_on_texts(allx)

        sequences = self.tokenizer.texts_to_sequences(allx)
        self.xtest_tweets = pad_sequences(sequences, maxlen=self.maxlen)
        self.ytest = np.asarray(ally)
        # self.test_idx = np.asarray(all_idx)

    def init_val(self, cleaned):
        """
            This is a function to initialize the training data.
            This function does not return anything. Instead, it sets instance
            fields to the appropriate values.

            :param allx: a list of input
            :param ally: a list of gold standard output
        """
        assert (hasattr(self, "tokenizer")), "This is for testing; init training!"
        allx = []
        ally = []
        all_idx = []


        for tweet in cleaned['tweet']:
            allx.append(tweet)
        for label in cleaned['label']:
            one_hot = [0, 0, 0, 0, 0, 0]
            one_hot[int(label)] += 1

            ally.append(one_hot)
        for idx in cleaned['idx']:
            all_idx.append([idx])

        all_prev_pos = cleaned['prev_POS']
        all_target_pos = cleaned['target_POS']
        all_next_pos = cleaned['next_POS']
        all_targets = cleaned['target']

        if params['feat_groups'] == '':
            self.val_feats = np.asarray(all_idx)
        elif params['feat_groups'] == 'pos_only':
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            self.val_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array), axis=1)
        elif params['feat_groups'] == 'all':
            idx_array = np.asarray(all_idx)
            all_prev_pos_array = np.asarray(all_prev_pos)
            all_target_pos_array = np.asarray(all_target_pos)
            all_next_pos_array = np.asarray(all_next_pos)
            all_targets_array = np.asarray(all_targets)
            self.val_feats = np.concatenate(
                (idx_array, all_prev_pos_array, all_target_pos_array, all_next_pos_array, all_targets_array), axis=1)

        self.tokenizer.fit_on_texts(allx)

        sequences = self.tokenizer.texts_to_sequences(allx)
        self.xval_tweets = pad_sequences(sequences, maxlen=self.maxlen)
        self.yval = np.asarray(ally)

        self.validation_data = ([self.xval_tweets, self.val_feats], self.yval)

    def _init_embeddings(self):
        """
            This is a function to read in the initial embedding weights,
            construct an embedding matrix, and construct an embedding layer
            from that matrix.

            No input is required; the path to the embeddings is contained
            within self.params.

            :return: the embedding layer
        """
        # Create a dictionary to store the embeddings that are to be read in
        embeds = {}
        embed_dim = None

        # Open the file from params that contains the initial embedding weights
        with open(self.params['initial_embed_weights']) as f:
            # Loop over the lines of the file. Each line contains a word
            # followed by a vector
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')

                # If this is the first word, get the number of dimensions
                if embed_dim is None:
                    embed_dim = len(coefs)
                    print("embed_dim set to " + str(embed_dim))

                # Normalize the embeddings to the unit sphere
                embeds[word] = coefs / np.linalg.norm(coefs)
        f.close()

        # Log useful information
        logging.info("Embedding dimension: %i" % (embed_dim))
        logging.info('Found {0} word vectors.'.format(len(embeds)))

        # Create a matrix to store the one-hots for each word
        # We add one to the size of the vocabulary for the UNK token
        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, embed_dim))

        # Create a dictionary to store OOV words 
        unk_dict = {}

        # Loop over the words in the vocabulary
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeds.get(word) # we use get here to avoid key errors from OOV words

            # If the word is in vocabulary, add its vector
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

            # If the word is OOV but has been seen before and is in the unk dictionary,
            # add the vector from there
            elif word in unk_dict:
                embedding_matrix[i] = unk_dict[word]

            # If the word is OOV and has not been seen before, create a new embedding
            else:
                # random initialization, see
                # https://github.com/bwallace/CNN-for-text-classification/blob/master/CNN_text.py
                unk_embed = np.random.random(embed_dim) * -2 + 1
                unk_dict[word] = unk_embed
                embedding_matrix[i] = unk_dict[word]

        #for i in unk_dict:
        #    print(i)

        #input("CHECK THIS SHIT OUT")

        # Create the embedding layer from the embedding matrix
        embedding_layer = Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                    output_dim=embed_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.maxlen,
                                    mask_zero=True)

        # Log the successful creation of the embedding layer and the number of unknown tokens
        logging.info("embedding layer completed")
        logging.info(str(len(unk_dict)) + " unknown words")

        return embedding_layer

    def init_bilstm(self):
        """
            This is a function to initialize the BiLISTM.
            :return:
        """

        # Declare the input ===== only tweets
        x_in = Input(shape=(self.xtrain_tweets.shape[1],), dtype='int32')

       # print(self.train_idx.shape[1]
       # input()
        #print(self.train_idx.shape)
        #feats_size = 3*self.params['pos_dims'] + 1
        feats_in = Input(shape=(self.train_feats.shape[1],), dtype='float32')

        #sent_in = Input(shape=(self.))
        # Declare the embedding layer
        embedding = self._init_embeddings()

        # Run the input through the embedding layer
        x_embed = embedding(x_in)

        # Create the BiLSTM
        bilstm = Bidirectional(LSTM(self.params['rnndim']))(x_embed)

        #####CONCAT


        
        # INSERT 
        #####
        concat = concatenate([bilstm, feats_in] ) #, mode='concat', concat_axis=-1)

        dense = Dense(6, activation="relu")(concat)

        # Create the dropout layer
        drop_out = Dropout(self.params['dropout'])(dense)

        # Create the dense layer for prediction
        toplayer = Dense(6, activation="softmax")(drop_out)

        # Construct the network
        self.model = Model(inputs=[x_in, feats_in], output=[toplayer])

        # Check parameters to determine if the embeddings are to be trained
        if not self.params["train_embed"]:
            self.model.layers[1].trainable = False

        # Construct an optimizer to pass to compile
        adam = optimizers.Adam(lr=params["lr"], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                               amsgrad=False)
        # Compile the model and print a summary.
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.model.summary()

    def train(self):
        """
            This is a function to train the model.
            :return:
        """
        MODEL_DIR = "checkpoints/"
        filepath = MODEL_DIR + self.params["sysid"] + "-{epoch:02d}-{loss:.4f}.hdf5"

        # Create a checkpoing to be updated during training
        checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True,
                                     mode="min")

        # Define early stopping
        callbacks_list = [checkpoint,
                          EarlyStopping(monitor="val_loss", patience=1)]

        xtrain = [self.xtrain_tweets, self.train_feats]

        print("Shape xtrain_tweets: " + str(self.xtrain_tweets.shape))
        print("Shape train_feats:" + str(self.train_feats.shape))

        # Train the model!
        # hist = self.model.fit(xtrain,
        #                       self.ytrain,
        #                       nb_epoch=self.params['nepoch'],
        #                       batch_size=self.params['batch_size'],
        #                       verbose=True,
        #                       validation_split=self.params['validation_split'],
        #                       callbacks=callbacks_list)

        print("Shape Validation_Data:" + str(len(self.validation_data)))
        print("Shape Val XTweets:" + str(self.xval_tweets.shape))
        print("Shape Val Feats: " + str(self.val_feats.shape))
        print("Shape Val Y: " + str(self.yval.shape))

        print("Shape XTest: " + (str(self.xtest_tweets.shape)))
        print("Shape YTest: " + str(self.ytest.shape))

        hist = self.model.fit(xtrain,
                       self.ytrain,
                       epochs=self.params['nepoch'],
                       batch_size=self.params['batch_size'],
                       verbose=True,
                       validation_data=self.validation_data,
                       callbacks=callbacks_list)
    def test(self):
        """
            This is a function to test the model
        :return:
        """

        # save_preds = open('./training/predictions/' + self.params['prefix'] + '.csv', 'w')
        #save_preds = open('./checkpoints/preds.csv', 'w')
        #writer = csv.writer(save_preds, delimiter=',')
        #writer.writerow(["Y_PRED", "Y_TRUE"])

        preds = self.model.predict([self.xtest_tweets, self.test_feats])
        pred_int = np.array([np.argmax(x) for x in preds])

        #save_preds.write("y_pred\ty_true\n")
        #for i in range(len(pred_int)):
           # save_preds.write(str(pred_int[i]) + "\t" + str(self.ytest[i]) + "\n")

        mic_f1 = f1_score(self.ytest, pred_int, labels=None, pos_label=1,
                                 average='micro', sample_weight = None)
        mac_f1 = f1_score(self.ytest, pred_int, labels=None, pos_label=1,
                                 average='macro', sample_weight = None)

        acc = accuracy_score(self.ytest, pred_int, normalize=True, sample_weight=None)

        print("Micro: " + str(mic_f1))
        print("Macro: " + str(mac_f1))
        print("Acc: " + str(acc))
        #for i in range(len(pred_int)):
           # row = [pred_int[i], self.ytest[i]]
           # print("Predicted: " + str(row[0]) + "; Gold: " + str(row[1]))



if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysid", default="Intention", help="System ID for checkpoint file saves")
    parser.add_argument("--model", help="Choose: bilstm or attn_rnn",
                        choices=("bilstm","attn_rnn"))
    parser.add_argument("--embed_dim", help="Embedding dim",
                        choices=(50,100,200,300),
                        type=int, default=None, action='store')
    parser.add_argument("--train_embed", help="train embeddings?", action='store_true')
    parser.add_argument("--rnndim", help="Projection dimension for RNN",
                        type=int, default=128, action='store')
    parser.add_argument("--dropout", help="Dropout rate",
                        type=float, default=0.2, action='store')
    parser.add_argument("--validation_split", help="% of training for validation",
                        type=float, default=0.2, action='store')
    parser.add_argument("--nepoch", type=int, default=20, action='store')
    parser.add_argument("--batch_size", type=int, default=64, action='store')
    parser.add_argument("--maxsentlen", type=int, default=60, action='store')
    parser.add_argument("--lr", type=float, default=0.001, action='store', help='Learning rate')
    parser.add_argument("--feat_groups", type=str, default="all", action='store', help='what model')
    args = parser.parse_args()
    params = vars(args)
    params['initial_embed_weights'] = "/Users/victormena/Development/twitter/vulgartwitter/glove.twitter.27B/glove.twitter.27B.200d.txt"

    print("Params = ", params)

    # Initialisze the model
    print("Cleaning the data.")
    clean_train, clean_test, clean_val, pos_dims = clean_input_intention("train_set.txt", "train_tagged.txt", "test_set.txt", "test_tagged.txt", "val_set.txt", "val_tagged.txt")
    print("Cleaning Done.")

    params['pos_dims'] = pos_dims

    model = BiLSTM(params)
    model.init_train(clean_train)
    model.init_test(clean_test)
    model.init_val(clean_val)
    model.init_bilstm()
    model.train()
    model.test()

from keras.utils.np_utils import to_categorical
from keras import optimizers
import pprint
from sklearn.metrics import mean_absolute_error
