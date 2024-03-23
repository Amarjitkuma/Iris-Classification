import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Add a title and a description
st.title('Iris Species Predictor')
st.write('This app predicts the species of an Iris flower based on the lengths and widths of its sepals and petals.')

# Define the feature input
st.write('Please enter the following parameters:')
sepal_length = st.number_input('Sepal Length (Typical range: 4.3 - 7.9 cm)',  value=5.0, step=1.0)
sepal_width = st.number_input('Sepal Width (Typical range: 2.0 - 4.4 cm)',  value=3.0, step=1.0)
petal_length = st.number_input('Petal Length (Typical range: 1.0 - 6.9 cm)',  value=4.0, step=1.0)
petal_width = st.number_input('Petal Width (Typical range: 0.1 - 2.5 cm)',  value=1.0, step=1.0)

# Create a dataframe from the inputs
features = pd.DataFrame([sepal_length, sepal_width, petal_length, petal_width]).T

# Scale the features
features = scaler.transform(features)
if st.button("Predict"):
# Make the prediction
    prediction = model.predict(features)

    # Display the prediction
    st.write(f'Prediction: {prediction[0]}')

    # Display the image of the predicted species
    if prediction[0] == 'Iris-setosa':
        st.image('iris-setosa.jpg', use_column_width=True)
    elif prediction[0] == 'Iris-versicolor':
        st.image('Iris_versicolor.jpg', use_column_width=True)
    elif prediction[0] == 'Iris-virginica':
        st.image('Iris-virginica.jpg', use_column_width=True)
