# How Cute is Your Pet?

This project features [an interactive web app](https://share.streamlit.io/kozodoi/pet_pawpularity/main/web_app.py) that estimates Pawpularity of custom pet photos. Pawpularity is a metric used by [PetFinder](https://petfinder.my/) to judge the pet's attractiveness, which translates to more clicks for the pet profile. You can simply upload your pet photo and learn the predicted score! The app uses one of three computer vision models developed within the [PetFinder Kaggle competition](https://www.kaggle.com/c/petfinder-pawpularity-score/overview).

![web_app](https://i.postimg.cc/90g241GT/Screen-2021-11-23-at-12-20-20.jpg)


## Project structure

The project has the following structure:
- `web_app.py`: Python script implementing the web app in Streamlit
- `results.csv`: table containing the recent image scoring results
- `codes/`: Python codes with data augmentation and model initialization functions
- `models/`: model configurations and out-of-fold predictions
