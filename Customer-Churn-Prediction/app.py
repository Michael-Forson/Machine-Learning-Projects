import streamlit as st
import joblib
import numpy as np

scalar = joblib.load("scaler.pkl")
model = joblib.load("loyalty_model.pkl")
st.title("Customer Loyalty Prediction")
st.write("This app predicts whether a customer will stay loyal or not based on their features.")
st.divider()
st.write("Please enter the customer features and press the button to predict.")
st.divider()
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)


tenure= st.number_input("Enter Tenure",min_value=0,max_value=100,value=10)
monthlycharge = st.number_input("Enter Monthly Charge",min_value=0.0,max_value=1000.0,value=50.0)
gender = st.selectbox("Enter the Gender",["Male","Female"]) 

st.divider()
predictbutton = st.button("Predict!")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0

    X=[age,gender_selected,tenure,monthlycharge]
    
    X1=np.array(X)
    X_array = scalar.transform([X1])
    prediction = model.predict(X_array)[0]

    predicted = "Loyal" if prediction == 0 else "Not Loyal"

    st.balloons()
    
    st.write(f"Predicted: {predicted}")

else :
    st.write("Please enter the values and use predict button")