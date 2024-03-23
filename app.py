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
sepal_length = st.number_input('Sepal Length',  value=0.0, step=1.0)
sepal_width = st.number_input('Sepal Width',  value=0.0, step=1.0)
petal_length = st.number_input('Petal Length',  value=0.0, step=1.0)
petal_width = st.number_input('Petal Width',  value=0.0, step=1.0)

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
        st.markdown("""
            # Iris Setosa
            Iris setosa, also known as the bristle-pointed iris, is a flowering plant with small, deep violet blue flowers and dark purple sepals. 
            It has narrow, stiff, green leaves and blooms in late spring. The plant can grow up to 24 inches tall and prefers full sun or part shade, 
            wet to mesic, neutral to slightly acid loam.
            """)
        st.image('iris-setosa.jpg', use_column_width=True)
    elif prediction[0] == 'Iris-versicolor':
        st.markdown("""
            # Iris Versicolor
            Iris versicolor, also known as the blue flag iris, northern blue flag, harlequin blue flag, larger blue flag, and poison flag, is a perennial herb native to North America. 
            It can grow up to three feet tall and has sword-like leaves and violet-blue flowers with yellow-based sepals. 
            The blue flag iris blooms from May to August, and its flowers can be white, yellow, blue, purple, or violet.
            """)
        st.image('Iris_versicolor.jpg', use_column_width=True)
    elif prediction[0] == 'Iris-virginica':
        st.markdown("""
            # Iris Virginica
            Iris virginica, also known as the Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial flowering plant native to central and eastern North America. 
            It's a wildflower that grows in the United States and Canada, typically in boggy areas with standing water.
            """)
        st.image('Iris-virginica.jpg', use_column_width=True)
