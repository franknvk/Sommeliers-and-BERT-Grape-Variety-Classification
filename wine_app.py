import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

@st.cache(allow_output_mutation = True)
def load_model(model):
    return keras.models.load_model(model)
model = load_model('my_model')

@st.cache(allow_output_mutation = True)
def load_data(model):
    data = pd.read_csv('data/2022_winemag_reviews.csv')
    data.dropna(subset = 'price',inplace = True)
    data.drop_duplicates(subset = 'title',inplace=True)
    return data
data = load_data('data/2022_winemag_reviews.csv')

mapping = {0: 'Cabernet Sauvignon',
 1: 'Chardonnay',
 2: 'Gamay',
 3: 'Merlot',
 4: 'Nebbiolo',
 5: 'Pinot Noir',
 6: 'Portuguese Red',
 7: 'Portuguese White',
 8: 'Rosé',
 9: 'Sangiovese',
 10: 'Sauvignon Blanc',
 11: 'Syrah',
 12: 'Tempranillo'}


st.markdown("<h1 style='text-align: center; color: grey;'>Welcome to the Bottle-O-Wine Selector</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: grey;'>The app to help you choose a wine based on specific tasting notes.</h6>", unsafe_allow_html=True)   
with st.form("my_form"):
    value = 'ex: blueberry, leather, rum'
    
    #user input for tasting notes
    input_text = st.text_input(label = "What tasting notes are you looking for?",value=value)
    price_text = st.text_input(label = "What is the maximum price you would like to spend?")
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        if input_text == value:
            st.write('Please choose some tasting notes!')
        elif price_text == '':
            st.write('Please pick a maximum price!')
        else:
            model = load_model('my_model')
            prediction = model.predict([input_text])
            encoded_wine_class = np.argmax(prediction)
            wine_type = mapping[encoded_wine_class]
            st.subheader(wine_type)
            st.caption('Here are some recs:')
            df = data.loc[(data['variety']==wine_type) & (data['price'] <= float(price_text))][['points','title','price']]
            df = df.sort_values(by='points',ascending=False).head(10)
            df