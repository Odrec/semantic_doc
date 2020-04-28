# semantic_doc
Testing some semantic models for classification

LDA: using topic modeling for classification. Partially based on https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28

Files:

edgar\_scrapper.py -- Downloads the document data from https://www.sec.gov/Archives/edgar/data/51143/ and saves the documents in pdf format with the name formatted in this way: \[name\_of\_folder\]\_\[name\_of\_file\]\_\[label(type)\].pdf. Over 7000 files gathered for 34 classes.

get\_data\_classes.py -- Extracts the content of the files and has functions to divide the data in training and testing sets.

data\_preprocess.py -- Has functions to preprocess the data for the models.

models.py -- Code for the training and testing of different models.

*Results:*

Classifiers scores for LDA bow model.

Score for classifier Logistic Regression SGD: 0.640555
Score for classifier SVM Huber: 0.640555
Score for classifier Random Forest: 0.640555

Classifiers scores for LDA bigram model.

Score for classifier Logistic Regression SGD: 0.640555
Score for classifier SVM Huber: 0.000112
Score for classifier Random Forest: 0.640555

Classifiers scores for LDA trigram model.

Score for classifier Logistic Regression SGD: 0.641057
Score for classifier SVM Huber: 0.640555
Score for classifier Random Forest: 0.000075

*Note:* There was a problem using the default BLAS library with the numpy and scipy where the multicore functionality was not working properly. Changing to using the openBLAS library fixed the multicore issues. Also, openBLAS is fairly optimized in comparison to the default one so the models now run much faster. The TF-IDF based model is still too heavy for single laptop.
