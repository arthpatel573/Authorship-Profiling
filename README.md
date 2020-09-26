# Authorship-Profiling
![python version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-blue?logo=python)

Study of authorship analytics deals with the grouping texts based on authors' stylistic choices of writing textual content. This type of study helps in distinguishing authors based on their sociological aspect. A formulated version of this task boils down to gender classification task, where gender needs to be predicted based on text. Many approaches for solving this task has been made in various ways by extracting meaningful features. 

So, experiments of different classification models developed for gender classification problem are performed considering authorship analysis. These classifiers perform classification tasks based on pre-processed text content from authors’ tweets and provides output as gender which is more likely to have written those texts.

## Setup

Unzip all_data/data.zip in the same folder.

Install all dependencies

```
pip install -r requirements.txt
```

## Experiments

Run each `preprocess-ml.ipynb' notebook and generate 'training_data_dl.csv' and 'testing_data_dl.csv'.

Once they are produced, run each experiment for deep learning classification model (found in 'tensorflow-expts.ipynb') on these files.

## Deploying the model

The final version of the model can be run by running `./scripts/train.py` script.

Then, follow the `./deploy.ipynb` file for the deploying this model on **AWS SageMaker**.

## Performance evaluation

For accuracy, a modified measure is developed which is normalized sum of all probabilities across all documents in each tweet
