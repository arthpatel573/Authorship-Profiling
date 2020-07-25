# Authorship-Profiling

Study of authorship analytics deals with the grouping texts based on authors' stylistic choices of writing textual content. This type of study helps in distinguishing authors based on their sociological aspect. A formulated version of this task boils down to gender classification task, where gender needs to be predicted based on text. Many approaches for solving this task has been made in various ways by extracting meaningful features. 

So, experiments of different classification models developed for gender classification problem are performed considering authorship analysis. These classifiers perform classification tasks based on pre-processed text content from authors’ tweets and provides output as gender which is more likely to have written those texts.

# Setup

Unzip all_data/data.zip in the same folder.

Install all dependencies

```
pip install -r requirements.txt
```

# Experiments

Run each `preprocess-ml.ipynb' notebook and generate 'training_data_dl.csv' and 'testing_data_dl'.

Once they are produced, run each experiment for deep learning classification model on these files.

# Performance evaluation

For accuracy, a modified measure is developed which is normalized sum of all probabilities across all documents in each tweet