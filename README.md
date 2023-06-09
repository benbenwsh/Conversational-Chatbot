# Conversational Chatbot
## Project Overview
A conversational chatbot built with a seq2seq LSTM model trained myself using the Cornell movie dialog corpus.

## Cool Features
- Uses a deep neural net (from Tensorflow) to predict the intent of the user input
  - If the confidence is high, it outputs a random response from a list of responses related to that intent
- Uses a seq2seq LSTM generative model to generate a response by predicting the next character
  - Used when the confidence of the intent recognition is low
- Detect intent for self-harm and respond with a message containing suicide prevention resources
- Profanity detection (usign better_profanity)

## Technologies Used
- Python modules: Tensorflow, nltk, etc.
- Frontend: HTML, CSS, JS
- Backend: Flask

## Structure of Codebase
- nlp
  - Contains the generative and the retrieval (intent detection) models, and the code for training and testing them
- static
  - Contains the css and js files
- template
  - Contains the html file
- app.py
  - The Flask app (entry point to the program)

## Installation and Setup
- Run app.py
- Go to the URL provided in the command line

## Screenshot
![chatbot](https://github.com/benbenwsh/Conversational-Chatbot/assets/38101123/63c0be77-63eb-45a6-b8e6-e67f05628176)
