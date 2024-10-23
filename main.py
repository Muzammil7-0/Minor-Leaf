import streamlit as st
import cv2 as cv
import numpy as np
import keras


label_name = ['Apple scab', 'Apple  rust', 'Apple healthy']

st.write("""The leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of more than 2000 different Apple leaves . For more information about the architecture, dataset, and training process, please refer to the code and Documents""")              

st.write("Please input only leaf Images of Apple. Otherwise, the model will not work perfectly.")

model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')


uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (250, 250)), axis=0)
    predictions = model.predict(normalized_image)
    st.image(image_bytes)
    if predictions[0][np.argmax(predictions)]*100 >= 80:
        st.write(f"Result is : {label_name[np.argmax(predictions)]}")
    else:st.write(f"Try Another Image")
