import streamlit as st
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd 
import pickle


model = tf.keras.models.load_model('model.h5')


## Loading the  encoders  and Scaler  : 
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb')as file:
    one_hot_encoder_geo  = pickle.load(file)
    
with open('sscaler.pkl','rb') as file:
    scaler  = pickle.load(file)
    
## Streamlit app 
st.title("Customer Churn Prediction")

# Create input fields
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600, step=1)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18,92)
tenure = st.slider('Tenure (in years)', 0,10)
balance = st.number_input('Account Balance', min_value=0.0, value=60000.0, step=1000.0)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4])
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)



input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## One-hot encode 'Geography'
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography']),
    index=[0]
)

## Label encode 'Gender'
gender_encoded = label_encoder_gender.transform([gender])[0]

# Create updated input_data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],  # Encoded
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Concatenate with one-hot encoded Geography
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba > 0.5:
    st.write('🔴 The customer is **likely to churn**.')
else:
    st.write('🟢 The customer is **not likely to churn**.')
