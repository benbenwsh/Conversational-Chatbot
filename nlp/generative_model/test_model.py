import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import pickle
from better_profanity import profanity

# Retrieving the pre-processed data
model_name = '148792_256_32_100'
num_samples, *_ = model_name.split('_')

with open(f'nlp/generative_model/Cornell Dataset/Preprocessed Data/{num_samples}.pickle', 'rb') as f:
    input_features_dict, target_features_dict, max_encoder_seq_length, max_decoder_seq_length = pickle.load(f)



# Convert the user response, which is a string, into a matrix using numpy, in order to feed into the model
def string_to_matrix(user_input, num_encoder_tokens):
    characters = list(user_input)

    user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for timestep, character in enumerate(characters):
        if character in input_features_dict:
            user_input_matrix[0, timestep, input_features_dict[character]] = 1.
    return user_input_matrix


# Converting the user response into a matrix and feed it into the seq2seq model to generate a response
def generate_response(user_input):
    model = load_model(f"nlp/generative_model/Cornell Dataset/{model_name}")  # Loading the trained model

    num_encoder_tokens = model.input[0].shape[2]

    # Building encoder
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder_inputs = model.input[0]
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)

    # Building decoder
    dimensionality = model.layers[3].units
    num_decoder_tokens = model.input[1].shape[2]
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_inputs = model.input[1]  # Decoder input layer (0 is Encoder input layer)
    decoder_state_input_hidden = Input(shape=(dimensionality,))
    decoder_state_input_cell = Input(shape=(dimensionality,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(model.input[1], initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # Dictionaries that have index as key and character as value
    reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
    reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())


    input_matrix = string_to_matrix(user_input, num_encoder_tokens)  # Convert user response into a matrix

    states_value = encoder_model.predict(input_matrix)  # The value of the cell state and the hidden state outputted by the encoder
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_features_dict['\t']] = 1.  # Temporarily store '\t' as the current character so that the decoder can predict the next character based on this information

    chatbot_response = ''

    stop_condition = False

    while not stop_condition:
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)  # Decoder takes the previous character and states value as input, and predicts the next character

        sampled_token_index = np.argmax(
            output_tokens[0, -1, :])  # The index with the highest probability in output_tokens
        sampled_token = reverse_target_features_dict[
            sampled_token_index]  # The character that is most likely to be the next character

        chatbot_response += sampled_token  # Append the character to produce a chatbot response

        # If the sampled_token is '\n', indicating it is the end of the sequence, or if the length of the prediction exceeds 100, stop appending to the sequence
        if (sampled_token == '\n' or len(chatbot_response) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[
            0, 0, sampled_token_index] = 1.  # Store the current character so that the decoder can predict the next character based on this information

        states_value = [hidden_state,
                        cell_state]  # Store the states value outputted by the decoder in order to feed into the decoder in the next iteration

    chatbot_response = chatbot_response.replace("\n", "").replace('\t', '')

    if profanity.contains_profanity(chatbot_response):  # If the response generated contains any swear word, it responds with a pre-defined statement instead
        chatbot_response = "I'm not sure I understand."

    return chatbot_response