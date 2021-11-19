##### PREPARATIONS

# libraries
import gc
import pickle
import glob
import os
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import torch

# custom libraries
sys.path.append('code')
from model import get_model
from augmentations import get_augs

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


##### HEADER

# title
st.title("How cute is your pet?")

# image cover
cover_image = Image.open(requests.get("https://storage.googleapis.com/kaggle-competitions/kaggle/25383/logos/header.png?t=2021-08-31-18-49-29", stream = True).raw)
st.image(cover_image)

# description
st.write("This app allows estimating the pawpularity score of custom pet photos. Pawpularity is a metric used by [PetFinder.my](https://petfinder.my/), which is a Malaysia’s leading animal welfare platform. Pawpularity serves as a proxy for the photo's attractiveness, which translates to more page views for the pet profile.")


##### PARAMETERS

# header
st.header('Score your own pet')

# photo upload
pet_image = st.file_uploader("1. Upload your pet photo.")
if pet_image is not None:
    
    # save image to folder
    image_path = pet_image.name
    with open(image_path, "wb") as f:
        f.write(pet_image.getbuffer())
    
    # display pet image
    with st.expander('Pet photo uploaded! Expand to check the photo.'):
        st.image(pet_image)      
        
# privacy toogle
choice = st.radio("2. Make the result public?", ["Yes. Others may see your pet photo.", "No. Scoring will be done privately."])

# model selection
model_name = st.selectbox(
    '3. Choose a model for scoring your pet.',
    ['EfficientNet B4'])


##### MODELING

# compute pawpularity
if st.button('Compute pawpularity'):
    
    # check if image is uploaded
    if pet_image is None:
        st.error('Please upload a pet image first.')
    else:

        # specify paths
        if model_name == 'EfficientNet B4':
            weight_path = 'https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/efficientnet.pth'
            model_path  = 'enet_b4/'
        elif model_name == 'SWIN Transformer':
            weight_path = 'https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/swin.pth'
            model_path  = 'swin/' 
            
        # download model weights
        if not os.path.isfile(model_path + 'pytorch_model.pth'):
            with st.spinner('Downloading model weights. This is done once and can take a minute...'):
                urllib.request.urlretrieve(weight_path, model_path + 'pytorch_model.pth', show_progress)

        # compute predictions
        with st.spinner('Computing prediction...'):

            # clear memory
            gc.collect()

            # load config
            config = pickle.load(open(model_path + 'configuration.pkl', 'rb'))

            # initialize model
            model = get_model(config, pretrained = model_path + 'pytorch_model.pth')
            model.eval()
            
            # define augmentations
            _, augs = get_augs(config)
            
            # process pet image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = augs(image = image)['image']
            image = torch.unsqueeze(image, 0)

            # compute prediction
            pred  = model(image)
            score = np.round(100 * pred.detach().numpy()[0][0], 2)
            st.write('**Predicted pawpularity:**  ', score)
            
            # about the scores
            st.write('**Note:** pawpularity ranges from 0 to 100. [Click here](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/274106) to read more about the metric.')

            # save results
            if choice == "Yes. Others may see your pet photo.":
                
                # load results
                results = pd.read_csv("results.csv")
                
                # save resized image
                example_img  = cv2.imread(image_path)
                example_img  = cv2.cvtColor(example_img, cv2.COLOR_BGR2RGB)
                example_path = "images/example_{}.jpg".format(len(results) + 1)
                cv2.imwrite(example_path, img = example_img)
                
                # write score to results
                row     = pd.DataFrame({"path": [example_path], "score": [score]})
                results = pd.concat([results, row], axis = 0)
                results.to_csv("results.csv",  index = False)
                
                # delete old image
                if os.path.isfile("images/example_{}.jpg".format(len(results) - 3)):
                    os.remove("images/example_{}.jpg".format(len(results) - 3))
                
            # clear memory and files
            del config, model, augs, image
            os.remove(image_path)
            gc.collect()
            

##### RESULTS

# header
st.header('Recent results')

# show recent pets
with st.expander('See recently scored pet photos'):
    
    # find most recent files
    results = pd.read_csv("results.csv")
    if len(results) > 3:
        results = results.tail(3).reset_index(drop = True)
                
    # display images in columns
    cols = st.columns(len(results))
    for col_idx, col in enumerate(cols):
        with col:
            st.write('Score: ', results['score'][col_idx])
            example_img = cv2.imread(results['path'][col_idx])
            example_img = cv2.resize(example_img, (256, 256))
            st.image(example_img) 