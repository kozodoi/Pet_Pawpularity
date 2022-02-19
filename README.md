# How Cute is Your Pet?

Top-4% solution to the [PetFinder Pawpularity Contest](https://www.kaggle.com/c/petfinder-pawpularity-score/overview) and [web app](https://share.streamlit.io/kozodoi/pet_pawpularity/main/web_app.py) for estimating cuteness of your pet photos.

![cover](https://github.com/kozodoi/Pet_Pawpularity/blob/main/app/header.png?raw=true-08-31-18-49-29)

- [Summary](#summary)
- [Demo app](#demo-app)
- [Project structure](#project-structure)
- [Working with the repo](#working-with-the-repo)


## Summary

Millions of stray animals around the world suffer on the streets or in shelters every day. [PetFinder](https://petfinder.my/) is Malaysia’s leading online animal welfare platform, featuring over 180,000 animals with 54,000 happily adopted. Understanding factors affecting the adoption speed is important to ensure that stray animals can find their homes faster.

This project uses Deep Learning to analyze pet images and metadata and predict the “Pawpularity” of pet photos. Pawpularity is a metric used by [PetFinder](https://petfinder.my/) to judge the pet's attractiveness, which translates to more clicks for the pet profile and faster adoption. The solution can be incorporated into AI tools that can help shelters to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements.

My solution is an ensemble of CNNs and transformer models implemented in `PyTorch`. Most architectures are initialized from the ImageNet weights and fine-tuned on [PetFinder](https://petfinder.my/) data using pet photos and meta-data. The solution reaches the top-4% of the Kaggle competition leaderboard and includes [an interactive web app](https://share.streamlit.io/kozodoi/pet_pawpularity/main/web_app.py) for scoring custom pet photos.


## Demo app

This project features [an interactive web app](https://share.streamlit.io/kozodoi/pet_pawpularity/main/web_app.py) that estimates Pawpularity of your pet photos. You can simply upload your pet photo and learn the predicted cuteness score! The app uses transformers and CNNs to make predictions.

![web_app](https://i.postimg.cc/vHr83Tkg/ezgif-com-gif-maker-3.gifg)


## Project structure

The project has the following structure:
- `app/`: codes implementing the interactive web app in Streamlit
- `code/`: `.py` main scripts with data, model, training and inference modules
- `convnext/`: dependencies for implementing `convnext` architecture
- `notebooks/`: `.ipynb` Colab-friendly notebooks with model training and ensembling
- `input/`: input data (images are not included due to size constraints and can be downloaded [here](https://www.kaggle.com/c/petfinder-pawpularity-score/data))
- `output/`: model configurations and figures exported from the notebooks


## Working with the repo

### Environment

To work with the repo, I recommend creating a virtual Conda environment from the `project_environment.yml` file:
```
conda env create --name petfinder --file project_environment.yml
conda activate petfinder
```

The file `requirements.txt` provides the list of dependencies for the web app.


### Reproducing solution

The solution can be reproduced in the following steps:
1. Run pretraining notebooks `pretraining_v1.ipynb` - `pretraining_v3.ipynb` to obtain pre-trained weights of base models.
2. Run training / fine-tuning notebooks `training_v01.ipynb` `pretraining_v65.ipynb` to obtain final model weights.
3. Run the inference notebook `inference.ipynb` to extract and ensemble predictions.

All training notebooks have the same structure and differ in model/data parameters. Different versions are included to ensure reproducibility. To understand the training process, it is sufficient to go through the `code/` folder and inspect one of the modeling notebooks.

More details are provided in the documentation within the scripts & notebooks.
