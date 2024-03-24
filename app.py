import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main():
    st.title("Clssification Problem")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification DL App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # Define the feature input
    sepal_length = st.text_input("Sepal Length","Type Here")
    sepal_width = st.text_input("Sepal Width","Type Here")
    petal_length = st.text_input("Petal Length","Type Here")
    petal_width = st.text_input("Petal Width","Type Here")
    if sepal_length != 'Type Here':
      sepal_length = float(sepal_length)
    if sepal_width != 'Type Here':
        sepal_width = float(sepal_width)
    if petal_length != 'Type Here':
        petal_length = float(petal_length)
    if petal_width != 'Type Here':
        petal_width = float(petal_width)
    
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
if __name__=='__main__':
    main()

