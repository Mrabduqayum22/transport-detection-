import streamlit as st
from fastai.vision.all import *
import plotly.express as px


st.title("The model that classify transports")

file = st.file_uploader("Upload the image", type=["png","jpeg","gif"])

if file:

    img = PILImage.create(file)

    st.image(img)

    model = load_learner("transport.pkl")

    pred, pred_id, prob = model.predict(img)

    st.success(f"Prediction result: {pred}")

    st.info(f"Probability: {prob[pred_id]*100:.1f}%")

    fig = px.bar(x=prob * 100, y=model.dls.vocab)

    st.plotly_chart(fig)

