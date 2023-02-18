import streamlit as st
import pandas as pd
import pickle
import nltk
import string
from nltk.corpus import stopwords
stopwords.words('english')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('Loving')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    x = []
    for i in text:
        if i.isalnum():
            x.append(i)
            
    y = []
    for i in x:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    z = []
    for i in y:
        z.append(ps.stem(i))
    return " ".join(z)

st.title('E-mail / SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    # 1] Preprocess
    transformed_sms = transform_text(input_sms)

    # 2] vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3] predict
    result = model.predict(vector_input)[0]

    # 4] Display

    if result == 1:
        st.header('Spam Message')

    else:
        st.header('Not a Spam Message')