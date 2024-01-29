import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# NLTK

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Wordcloud

from wordcloud import WordCloud

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Evaluation Metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#for exporting model
import pickle

# Define a function to preprocess the text data
def preprocess_text(text):
    """
    Preprocess text data by removing stop words, punctuation, and lemmatizing the words.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """

    stop_words = set(stopwords.words('english'))

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = text.split()

    # Remove stop words
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Return the preprocessed text, the text should be in string form
    return ' '.join(tokens) # returns the preprocessed text by joining the tokens together with a space between each token

# Read the training and validation data
df_train = pd.read_csv("C:\\Users\\deela\\Downloads\\archive (3)\\train.txt",delimiter=';', names=['text','label'])
df_val = pd.read_csv("C:\\Users\\deela\\Downloads\\archive (3)\\val.txt", delimiter=';', names=['text','label'])

# Combine the training and validation data
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

# Convert the labels to binary (positive/negative)
df['label'].replace(to_replace=['surprise', 'joy', 'love'], value=1, inplace=True)
df['label'].replace(to_replace=['anger', 'sadness','fear'], value=0, inplace=True)

# Preprocess the text data
df['text'] = df['text'].apply(preprocess_text)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)


df.dropna(subset=['text'], inplace=True)

# Create a CountVectorizer object
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit the CountVectorizer object to the training data
vectorizer.fit(X_train)

# Transform the training and test data into vectors
X_train_vectors = vectorizer.transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Create a RandomForestClassifier object
classifier = RandomForestClassifier()

# Train the RandomForestClassifier object on the training data
classifier.fit(X_train_vectors, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test_vectors)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy) 

# Define a function to make predictions on new data
def predict_sentiment(text):
    """
    Make a prediction about the sentiment of a piece of text.

    Args:
        text (str): The text to be analyzed.

    Returns:
        str: The predicted sentiment, either "positive" or "negative".
    """

    # Preprocess the text data
    text = preprocess_text(text)

    # Transform the text into a vector
    text_vector = vectorizer.transform([text])

    # Make a prediction
    prediction = classifier.predict(text_vector)[0]

    # Return the predicted sentiment
    if prediction == 0:
        return 'negative'
    elif prediction == 1:
        return 'positive'
    else:
        return 'neutral'


pickle.dump(classifier,open('model.pkl','wb'))