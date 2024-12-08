import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
data = pd.read_csv("C:/Users/Omkar/Desktop/Spam Email Detection/email library.csv")
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

# Function to predict the category of a message
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]

# Streamlit UI
st.header('Email Spam Detection')

input_mess = st.text_input('Enter your Email here')
if st.button('Check'):
    output = predict(input_mess)
    st.markdown(f'The email is: **{output}**')
