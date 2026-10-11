# CodeAlpha_FAQChatbot

CodeAlpha AI Internship - Task 2: Chatbot for FAQs

A chatbot that answers customer questions for an online store by finding the most similar FAQ.

## How it works
1. FAQ questions and answers are stored in the code
2. Text is cleaned with NLTK: lowercase, remove punctuation, tokenize, remove stop-words, lemmatize
3. Questions are converted to numbers with TF-IDF
4. The user's question is matched to the closest FAQ using cosine similarity
5. The best answer is shown as the chatbot's reply

## Run
pip install scikit-learn nltk
python chatbot.py

## Tech
Python, NLTK, scikit-learn
