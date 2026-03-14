import streamlit as st
import joblib
import pandas as pd
import numpy as np


vectorizer=joblib.load("vectorizer_model.pkl")
model=joblib.load("sentiment_model.pkl")

st.set_page_config(layout="wide")
st.sidebar.image(r"C:\Users\Krishna\Downloads\flag jpb")
st.sidebar.title("About Project")
st.sidebar.write("Objective of this project is to predict sentiment(neg/Pos) of food review")
st.sidebar.title("Libraries")
st.sidebar.markdown("""
-numpy
-pandas
-sklearn
""")

st.sidebar.title("Cloud")
st.sidebar.markdown("streamlit")
st.sidebar.title("Contact")
st.sidebar.markdown("9999999999")
st.markdown("""
<div style="
background-color: orange;
padding: 20px;
border-radius: 10px;
text-align: center;
font-size: 35px;
font-weight: bold;
color: white;">
Food Sentiment Analysis
</div>
""", unsafe_allow_html=True)
st.write("\n")
col1,col2=st.columns([.4,.6])
with col1:
    review=st.text_input("**Enter Review**")
    if st.button("Predict"):
        x_test=vectorizer.transform([review])
        pred=model.predict(x_test)
        prob=model.predict_proba(x_test)
        if pred[0]==0:
            st.error("**Sentiment = Negative👎**")
            st.warning(f"Confidance Score ={prob[0][0]:.2f}")
        else:
            st.error("**Sentiment = Positive👍**")
            st.warning(f"Confidance Score ={prob[0][1]:.2f}")
with col2:
    st.header("Predict Bulk Reviews from CSV")
    file=st.file_uploader("**Select a csv file**",type=["csv","txt"])
    if file:
        df=pd.read_csv(file,header=None,names=["Review"])
        placeholder=st.empty()
        placeholder.dataframe(df)
        if st.button("Bulk Prediction"):
            x_test=vectorizer.transform(df.Review)
            pred=model.predict(x_test)
            prob=model.predict_proba(x_test)
            sentiment=["Positive" if i==1 else "Negative" for i in pred]
            df['Sentiment']=sentiment
            df['Confidence']=np.max(prob,axis=1)
            placeholder.dataframe(df)
            
        
    
            