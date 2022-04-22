import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from better_profanity import profanity

import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle
import string

from nlp.generative_model.test_model import generate_response

# Loading 'intents.json'
with open('nlp/retrieval_model/intents.json') as file:
    data = json.load(file)

# Loading preprocessed data (4 lists)
with open('nlp/retrieval_model/data.pickle', 'rb') as f:
    distinct_tokens, distinct_intents, training_data, output_data = pickle.load(f)

# Making neural net
net = tflearn.input_data(shape=[None, len(training_data[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output_data[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Loading previously trained model
model.load('nlp/retrieval_model/model.tflearn')

def bag_of_words(user_input, distinct_tokens):
    bag = [0 for _ in range(len(distinct_tokens))]

    # Tokenisation and stemming
    tokens = nltk.word_tokenize(user_input)
    tokens = [stemmer.stem(token.lower()) for token in tokens]
    print(tokens)

    for token in tokens:
        for i, distinct_token in enumerate(distinct_tokens):
            if distinct_token == token:
                bag[i] = 1

    return np.array(bag)


def respond(user_input):
    if profanity.contains_profanity(user_input):
        return 'I won\'t respond to that'

    if user_input[-1] not in string.punctuation:  # If the last letter in reply is not in the list of punctuations, add a fullstop at the end of the sentence.
        user_input = user_input + "."

    results = model.predict([bag_of_words(user_input, distinct_tokens)])[0]     # Our model makes prediction based on the user input

    results_index = np.argmax(results)      # Returns the index of the greatest value in the 'results' list, as the model predicts the highest probability for that intent
    results_intent = distinct_intents[results_index]       # Returns the intent of that index

    if results[results_index] > 0.99:        # Return a response from that intent only if the confidence the model has is greater than 70%
        for intent in data['intents']:
            if intent['tag'] == results_intent:
                responses = intent['responses']
        return random.choice(responses)     # Pick a random response from the list of responses of that specific intent
    else:
        return generate_response(user_input)     # If the confidence is lower than 99%, generate response from the generative model