import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

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


#df.dropna(subset=['text'], inplace=True)

# Create a CountVectorizer object
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit the CountVectorizer object to the training data
vectorizer.fit(X_train)

#Transform the training and test data into vectors
#X_train_vectors = vectorizer.transform(X_train)
#X_test_vectors = vectorizer.transform(X_test)

# Create a RandomForestClassifier object
#classifier = RandomForestClassifier()

# Train the RandomForestClassifier object on the training data
#classifier.fit(X_train_vectors, y_train)

# Make predictions on the test data
#y_pred = classifier.predict(X_test_vectors)

# Evaluate the model performance
#accuracy = accuracy_score(y_test, y_pred)
#print('Accuracy:', accuracy) 




#load the model
classifier=pickle.load(open('model.pkl','rb'))


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

# Example usage:
#print(predict_sentiment('i am happy today'))


##################################################################

from flask import Flask, request, json
import os
from flask_cors import CORS
import smtplib
from email.mime.text import MIMEText
from flask import jsonify



app = Flask(__name__)

CORS(app)









#@app.route('/message', methods=['GET'])
#def test():
 #   query = request.args.get('message')
    
  #  print(query)
   # sentiment_result = predict_sentiment(query)
    
    #tmp = json.dumps(sentiment_result)
   # print(tmp)
    #return json.loads(tmp)

@app.route('/message', methods=['GET'])
def test():
    query = request.args.get('message')
    
    if query is not None:
        print(query)
        sentiment_result = predict_sentiment(query)
        print(sentiment_result)
        return jsonify({"sentiment": sentiment_result})
    else:
        return jsonify({"error": "Invalid input. 'message' parameter is missing or set to None."}), 400



'''def send_email(recipient, body):
    msg = MIMEText(body)
    sender = 'deelanlasrado44@gmail.com'
    password = 'Deelanlasrado2003'
    msg['Subject'] = 'Help Needed'

    with smtplib.SMTP('smtp-mail.outlook.com', 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        print("Message sent!")'''

#@app.route('/alert', methods=['POST'])
#def alert():
    #data = request.get_json()
    #receipient = data['email']
    #documents = data['data']
    #name = data['name']

    #journals = []
    #for document in documents :
     #   journals.append(document['text'])

    #response = sentiment_chain({"person_name" : name, "journal_entries": journals})

   # tmp = json.dumps(response)
    #output = json.loads(tmp)

    #body = json.loads(output['text'])['body']

    #send_email(receipient, body)

    #return data



#####################################################################################################################


def send_email(recipient, body):
    msg = MIMEText(body)
    sender = 'deelan.cs21@sahyadri.edu.in'
    password = 'deelan2003'
    msg['Subject'] = 'Help Needed'

    with smtplib.SMTP('smtp-mail.outlook.com', 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        print("Message sent!")

@app.route('/alert', methods=['POST'])
def alert():
    data = request.get_json()
    recipient = data['email']
    documents = data['data']
    name = data['name']

    journals = []
    for document in documents:
        journals.append(document['text'])

    #sentiment_result = predict_sentiment(query)
    #print(sentiment_result)
    #return jsonify({"sentiment": sentiment_result})
    response = {"person_name": name, "journal_entries": journals}

    tmp = json.dumps(response)
    output = json.loads(tmp)

    body = f"""
            Summary line: You are a helpful mental health bot. Analyze the given journals by {name} and seek insights.
            Explain the situation to the writer's loved ones in a friendly manner and also suggest ways his loved ones can help him.

            {', '.join(journals)}
            """
    print(body)
    print(data)
    send_email(recipient, body)

    return data


    #tmp = json.dumps(response)
    #output = json.loads(tmp)

    #body = json.loads(output['text'])['body']

    #send_email(recipient, body)

    #return data



if __name__ == '__main__':
    app.run(debug=True)