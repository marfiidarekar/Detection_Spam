import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Function to train and save the model
def train_and_save_model():
    # Load the dataset
    data = pd.read_csv("email library.csv")
    
    # Clean the dataset
    data.drop_duplicates(inplace=True)
    
    # Replace categories 'ham' and 'spam' with 'Not Spam' and 'Spam'
    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
    
    # Separate features and target
    mess = data['Message']
    cat = data['Category']
    
    # Split the data into training and test sets
    mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)
    
    # Create a CountVectorizer object
    cv = CountVectorizer(stop_words='english')
    features = cv.fit_transform(mess_train)
    
    # Initialize the Naive Bayes model
    model = MultinomialNB()
    
    # Train the model
    model.fit(features, cat_train)
    
    # Save the trained model and vectorizer
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(cv, 'vectorizer.pkl')

# Function to load the trained model
def load_model():
    model = joblib.load('spam_model.pkl')
    cv = joblib.load('vectorizer.pkl')
    return model, cv

# If the model is not trained yet, train and save it
import os
if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
    train_and_save_model()

# Load the trained model and vectorizer
model, cv = load_model()

# Function to predict the category of a message
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# Streamlit UI
st.header('Email Spam Detection')

# File uploader for custom dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset uploaded successfully!")
    st.dataframe(data.head())
else:
    st.write("You can upload your own dataset to test the model.")

# Text input for testing the model
input_mess = st.text_input('Enter your Email here')

# When the "Check" button is clicked
if st.button('Check'):
    if input_mess:
        output = predict(input_mess)
        st.markdown(f'The email is: **{output}**')
    else:
        st.warning("Please enter an email to check!")
