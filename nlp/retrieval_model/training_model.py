import nltk     # Natural Language Toolkit: A suite of libraries for symbolic and statistical natural language processing for English
from nltk.stem.lancaster import LancasterStemmer        # A word stemmer based on the Lancaster (Paice/Husk) stemming algorithm.
stemmer = LancasterStemmer()

import numpy as np      # Library for the handling of large, multi-dimensional arrays and matrices
import tflearn      # TensorFlow deep learning library
import tensorflow as tf     # Library for machine learning and artificial intelligence
import json
import pickle       # Module for serialising and de-serialising Python object structures

# Loading JSON file
with open('nlp/retrieval_model/intents.json') as file:
    data = json.load(file)


# Preprocessing
distinct_tokens = []      # All distinct tokens (words) sorted
distinct_intents = []         # Distinct intents (tags) sorted
tokens_training = []       # 2D list of tokens of the sentence ('pattern')
intents_output = []     # Intents (tags) corresponding to each sentence ('pattern') in tokens_x

# Put data from 'intents.json' into the 4 lists
for intent in data['intents']:      # Loop through every intent
    for pattern in intent['patterns']:      # Loop through every sentence for each intent
        tokens = nltk.word_tokenize(pattern)        # Tokenisation, returns a list of tokenised copy of a sentence ('pattern') using NLTK's tokeniser
        distinct_tokens.extend(tokens)      # Add all tokens into 'distinct_token', which will be refined later

        tokens_training.append(tokens)      # Append a list of tokens, representing the sentence ('pattern') to 'tokens_training'
        intents_output.append(intent['tag'])        # Append the intent corresponding to this sentence to 'intents_output'

        if intent['tag'] not in distinct_intents:
            distinct_intents.append(intent['tag'])      # Append an intent (tag) which is not in the list yet to create a list of distinct intents

distinct_tokens = [stemmer.stem(token.lower()) for token in distinct_tokens if token != "," and token != "."]        # Stemming and lowercasing the words
distinct_tokens = sorted(list(set(distinct_tokens)))        # Sort and remove duplicate words in distinct_tokens
distinct_intents = sorted(distinct_intents)     # Sort distinct_intents
print(distinct_tokens)

# Making bag of words
training_data = []       # Numpy array of training data to be fed into the training model
output_data = []     # Numpy array of output data to be fed into the training model

for x, tokens in enumerate(tokens_training):        # Loop through each sentence
    bag = []

    tokens = [stemmer.stem(token) for token in tokens]        # Stemming

    # Loop through all the words (tokens) in each sentence.
    for distinct_token in distinct_tokens:
        if distinct_token in tokens:
            bag.append(1)       # If a word is present in the sentence, append a 1 in that position
        else:
            bag.append(0)       # Otherwise, append a 0

    # output_row is a one-hot vector, where a 1 in a position represents the correct intent for the respective user input
    output_row = [0 for _ in range(len(distinct_intents))]
    output_row[distinct_intents.index(intents_output[x])] = 1

    training_data.append(bag)
    output_data.append(output_row)

training_data = np.array(training_data)
output_data = np.array(output_data)


# Saving the preprocessed data
with open('nlp/retrieval_model/data.pickle', 'wb') as f:
    pickle.dump((distinct_tokens, distinct_intents, training_data, output_data), f)


tf.compat.v1.reset_default_graph()

# Creating neural net
net = tflearn.input_data(shape=[None, len(training_data[0])])        # Input layer
net = tflearn.fully_connected(net, 8)       # Hidden layer
net = tflearn.fully_connected(net, 8)       # Hidden layer
net = tflearn.fully_connected(net, len(output_data[0]), activation='softmax')        # Output layer
net = tflearn.regression(net)       # Regression layer (for reducing loss)

model = tflearn.DNN(net)        # Deep Neural Network model (Feedforward neural network)


# Train and save model
history = model.fit(training_data, output_data, n_epoch=1000, batch_size=8, show_metric=True)       # Training the model

model.save('nlp/retrieval_model/model.tflearn')     # Saving the trained model