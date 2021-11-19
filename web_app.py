# libraries
import gc
import os
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# custom libraries
#sys.path.append('code')
#from model import get_model
#from tokenizer import get_tokenizer

# download with progress bar
mybar = None
def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)

# title
st.title("How cute is your pet?")

# image cover
cover_image = Image.open(requests.get("https://www.petfinder.my/images/cuteness_meter.jpg", stream = True).raw)
st.image(cover_image)

# description
st.write("This app allows estimating the pawpularity score of custom pet photos. Pawpularity is a metric used by [PetFinder.my](https://petfinder.my/), which is a Malaysia’s leading animal welfare platform. Pawpularity serves as a proxy for the photo's attractiveness, which translates to more page views for the pet profile.")



# photo upload
pet_image = st.file_uploader("1. Upload your pet photo.")

# model selection
model_name = st.selectbox(
    '2. Choose model for scoring your pet.',
    ['EfficientNet B5'])

# compute readability
if st.button('Compute pawpularity'):

    # specify paths
    if model_name == 'EfficientNet B5':
        weight_path = 'https://github.com/kozodoi/Kaggle_Readability/releases/download/0e96d53/weights_v59.pth'
    elif model_name == 'DistilRoBERTa':
        weight_path = 'https://github.com/kozodoi/Kaggle_Readability/releases/download/0e96d53/weights_v47.pth'

    # download model weights
    if not os.path.isfile(folder_path + 'pytorch_model.bin'):
        with st.spinner('Downloading model weights. This is done once and can take a minute...'):
            urllib.request.urlretrieve(weight_path, 'pytorch_model.bin', show_progress)

    # compute predictions
    with st.spinner('Computing prediction...'):

        # clear memory
        gc.collect()

        # load config
        config = pickle.load(open(folder_path + 'configuration.pkl', 'rb'))
        config['backbone'] = folder_path

        # initialize model
        model = get_model(config, name = model_name.lower(), pretrained = folder_path + 'pytorch_model.bin')
        model.eval()

        # clear memory
        del tokenizer, text, config
        gc.collect()

        # compute prediction
        if input_text != '':
            prediction = model(inputs, masks, token_type_ids)
            prediction = prediction['logits'].detach().numpy()[0][0]

        # clear memory
        del model, inputs, masks, token_type_ids
        gc.collect()

        # print output
        st.write('**Predicted pawpularity score:** ', np.round(prediction, 4))

# about the scores
st.write('**Note:** pawpularity ranges from 0 to 100. [Click here](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274106) to read more about the metric.')