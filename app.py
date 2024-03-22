
# import subprocess

# # Path to the requirements.txt file
# requirements_file = "requirements.txt"

# # Run pip install command
# subprocess.run(["pip", "install", "-r", requirements_file], check=True)

import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image


pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(sepal_length,sepal_width,petal_length,petal_width):

    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: SepalLength
        in: query
        type: number
        required: true
      - name: SepalWidth
        in: query
        type: number
        required: true
      - name: PetalLength
        in: query
        type: number
        required: true
      - name: PetalWidth
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction=classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    res = prediction[0]
    i,k,Max = 0,0,max(res)
    output = ""
    for r in res:
        if r == Max:
            k = i
        i += 1    
    if k == 0: output = "Iris - Setosa"
    elif k == 1: output = "Iris - Versicolor"
    else: output = "Iris - Virginica"    
    return output



def main():
    st.title("Clssification Problem")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification DL App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
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


    result=""
    if st.button("Predict"):
        result=predict_note_authentication(sepal_length,sepal_width,petal_length,petal_width)
        st.success('The output is {}'.format(result))
        if result == "Iris - Setosa":
            st.markdown("""
            # Iris Setosa
            Iris setosa, also known as the bristle-pointed iris, is a flowering plant with small, deep violet blue flowers and dark purple sepals. 
            It has narrow, stiff, green leaves and blooms in late spring. The plant can grow up to 24 inches tall and prefers full sun or part shade, 
            wet to mesic, neutral to slightly acid loam.
            """)
            setosa_image_path = "iris-setosa.jpg"
            st.image(setosa_image_path, caption="Iris Setosa Flower")
        elif result == "Iris - Versicolor":
            st.markdown("""
            # Iris Versicolor
            Iris versicolor, also known as the blue flag iris, northern blue flag, harlequin blue flag, larger blue flag, and poison flag, is a perennial herb native to North America. 
            It can grow up to three feet tall and has sword-like leaves and violet-blue flowers with yellow-based sepals. 
            The blue flag iris blooms from May to August, and its flowers can be white, yellow, blue, purple, or violet.
            """)
            setosa_image_path = "Iris_versicolor.jpg"
            st.image(setosa_image_path, caption="Iris Versicolor Flower")
        else:
            st.markdown("""
            # Iris Virginica
            Iris virginica, also known as the Virginia blueflag, Virginia iris, great blue flag, or southern blue flag, is a perennial flowering plant native to central and eastern North America. 
            It's a wildflower that grows in the United States and Canada, typically in boggy areas with standing water.
            """)
            setosa_image_path = "Iris-virginica.jpg"
            st.image(setosa_image_path, caption="Iris Verginica Flower")
            
    
    

if __name__=='__main__':
    main()



