import streamlit as st
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img


st.markdown('<h1 style="color:black;">Vgg 16 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>',
            unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> implant cu fractura - 0,  implant fara fractura - 1</h3>', unsafe_allow_html=True)


# background image to streamlit
@st.experimental_memo()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.webp')

# load the trained model.
model = tf.keras.models.load_model('vgg_16_-saved-model-02-acc-0.86.hdf5')

upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    image = load_img(upload, target_size=(150, 150))
    image = np.asarray(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    c1.header('Input Image')
    c1.image(image, clamp=True)

    # prediction on model
    preds = model.predict(image)
    pred_classes = np.argmax(preds, axis=1)
    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(pred_classes[0])
