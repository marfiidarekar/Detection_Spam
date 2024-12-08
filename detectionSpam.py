import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
data = pd.read_csv("C:/Users/Omkar/Desktop/Spam Email Detection/email library.csv")
#print(data.head(5))

# Check dataset shape and clean it
#print(data.shape)
data.drop_duplicates(inplace=True)
#print(data.shape)  

# Check for missing values
#print(data.isnull().sum())

# Replacing categories 'ham' and 'spam' with 'Not Spam' and 'Spam'
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
#print(data.head(5))

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

# Test the model
features_test = cv.transform(mess_test)
#print(model.score(features_test,cat_test))

#predict data

def predict(message):
    input_message = cv.transform([message]).toarray()

    result=model.predict(input_message)
    return result

st.header('Email Spam Detection')



input_mess = st.text_input_('Enter your Email here')
if st.button('check'):
    output = predict(input_mess)
    st.markdown=output