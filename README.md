# semantic_doc
Testing some semantic models for classification

LDA: using topic modeling for classification. Partially based on https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28

Files:

edgar\_scrapper.py -- Downloads the document data from https://www.sec.gov/Archives/edgar/data/51143/ and saves the documents in pdf format with the name formatted in this way: \[name\_of\_folder\]\_\[name\_of\_file\]\_\[label(type)\].pdf. Over 7000 files gathered for 34 classes.

get\_data\_classes.py -- Extracts the content of the files and has functions to divide the data in training and testing sets.

data\_preprocess.py -- Has functions to preprocess the data for the models.

models.py -- Code for the training and testing of different models.
