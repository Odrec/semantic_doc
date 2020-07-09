#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:29:59 2020

@author: odrec
"""
from gensim import models
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn import linear_model

from os.path import isfile

import numpy as np
np.__config__.show()
import nltk
nltk.download('wordnet')

import data_preprocess as dp

#Trains a LDA model using the bow corpus.
def lda_model(corpus, dictionary, number_of_topics=20, save_path='saved_models/lda_bow'):
    if not isfile(save_path):
        lda_model = models.LdaMulticore(corpus, num_topics=number_of_topics, id2word=dictionary, passes=2, workers=2)
        lda_model.save(save_path)
    else: 
        lda_model = models.LdaMulticore.load(save_path)
    return lda_model

#Trains a LDA model using the tf-idf corpus. TAKES TOO LONG
def lda_model_tfidf(tfidf_corpus, dictionary, number_of_topics=20, save_path='saved_models/lda_tfidf'):
    if not isfile(save_path):
        lda_model_tfidf = models.LdaMulticore(tfidf_corpus, num_topics=number_of_topics, \
                                                     id2word=dictionary, passes=2, workers=4)
        lda_model_tfidf.save(save_path)
    else: 
        lda_model_tfidf = models.LdaMulticore.load(save_path)
    return lda_model_tfidf

#Transforms the topics into features for training and testing
def get_topics_features(model, corpus, number_of_topics=20):
    doc_vecs = []
    for i,f in enumerate(corpus):
        top_topics = model.get_document_topics(corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(number_of_topics)]
        doc_vecs.append(topic_vec)
    return doc_vecs
        
def train_classifiers(train_vecs, train_labels, typ='bow'):
    X = np.array(train_vecs)
    y = np.array(train_labels)
    
    kf = KFold(5, shuffle=True, random_state=42)
    cv_rf_f1, cv_lrsgd_f1, cv_svcsgd_f1,  = [], [], []
    cv_rf_ac, cv_lrsgd_ac, cv_svcsgd_ac,  = [], [], []
    y_pred_sgd, y_pred_sgh, y_pred_rf, = [], [], []
    
    for train_ind, val_ind in kf.split(X, y):
        # Assign CV IDX
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]
        
        # Scale Data
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_val_scale = scaler.transform(X_val)
    
        # Logisitic Regression
#        lr = LogisticRegression(
#            max_iter=4000,
#            class_weight= 'balanced',
#            solver='newton-cg',
#            fit_intercept=True
#        ).fit(X_train_scale, y_train)
#    
#        y_pred = lr.predict(X_val_scale)
#        cv_lr_f1.append(f1_score(y_val, y_pred, average='weighted'))
        
        # Logistic Regression SGD
        sgd = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            loss='log',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)
        
        y_pred_sgd.append(sgd.predict(X_val_scale))
        cv_lrsgd_f1.append(f1_score(y_val, y_pred_sgd[-1], average='macro'))
        cv_lrsgd_ac.append(accuracy_score(y_val, y_pred_sgd[-1]))
        
        # SGD Modified Huber
        sgd_huber = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)
        
        y_pred_sgh.append(sgd_huber.predict(X_val_scale))
        cv_svcsgd_f1.append(f1_score(y_val, y_pred_sgh[-1], average='macro'))
        cv_svcsgd_ac.append(accuracy_score(y_val, y_pred_sgh[-1]))
        
        # Random Forest
        rf = RandomForestClassifier(
            class_weight='balanced'
        ).fit(X_train_scale, y_train)
        
        y_pred_rf.append(rf.predict(X_val_scale))
        cv_rf_f1.append(f1_score(y_val, y_pred_rf[-1], average='macro'))
        cv_rf_ac.append(accuracy_score(y_val, y_pred_rf[-1]))
        
    y_pred_sgd_final = [item for sublist in y_pred_sgd for item in sublist]
    y_pred_sgh_final = [item for sublist in y_pred_sgh for item in sublist]
    y_pred_rf_final = [item for sublist in y_pred_rf for item in sublist]
    
#    print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
    print(f'SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}',typ)
    print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}',typ)
    print(f'Random Forest Val f1: {np.mean(cv_rf_f1):.3f} +- {np.std(cv_rf_f1):.3f}',typ)
    print("\n")
    print(f'SGD Val acc: {np.mean(cv_lrsgd_ac):.3f} +- {np.std(cv_lrsgd_ac):.3f}',typ)
    print(f'SVM Huber Val acc: {np.mean(cv_svcsgd_ac):.3f} +- {np.std(cv_svcsgd_ac):.3f}',typ)
    print(f'Random Forest Val acc: {np.mean(cv_rf_ac):.3f} +- {np.std(cv_rf_ac):.3f}',typ)
    print("\n")
    print("Precision (micro) SGD: %f" % precision_score(y, y_pred_sgd_final, average='micro'),typ)
    print("Recall (micro) SGD:    %f" % recall_score(y, y_pred_sgd_final, average='micro'),typ)
    print("F1 score (micro) SGD:  %f" % f1_score(y, y_pred_sgd_final, average='micro'),typ, end='\n\n')
    print("Precision (macro) SGD: %f" % precision_score(y, y_pred_sgd_final, average='macro'),typ)
    print("Recall (macro) SGD:    %f" % recall_score(y, y_pred_sgd_final, average='macro'),typ)
    print("F1 score (macro) SGD:  %f" % f1_score(y, y_pred_sgd_final, average='macro'),typ, end='\n\n')
    print("Precision (weighted) SGD: %f" % precision_score(y, y_pred_sgd_final, average='weighted'),typ)
    print("Recall (weighted) SGD:    %f" % recall_score(y, y_pred_sgd_final, average='weighted'),typ)
    print("F1 score (weighted) SGD:  %f" % f1_score(y, y_pred_sgd_final, average='weighted'),typ)
    print("\n")
    print("Precision (micro) SVM Huber: %f" % precision_score(y, y_pred_sgh_final, average='micro'),typ)
    print("Recall (micro) SVM Huber:    %f" % recall_score(y, y_pred_sgh_final, average='micro'),typ)
    print("F1 score (micro) SVM Huber:  %f" % f1_score(y, y_pred_sgh_final, average='micro'),typ, end='\n\n')
    print("Precision (macro) SVM Huber: %f" % precision_score(y, y_pred_sgh_final, average='macro'),typ)
    print("Recall (macro) SVM Huber:    %f" % recall_score(y, y_pred_sgh_final, average='macro'),typ)
    print("F1 score (macro) SVM Huber:  %f" % f1_score(y, y_pred_sgh_final, average='macro'),typ, end='\n\n')
    print("Precision (weighted) SVM Huber: %f" % precision_score(y, y_pred_sgh_final, average='weighted'),typ)
    print("Recall (weighted) SVM Huber:    %f" % recall_score(y, y_pred_sgh_final, average='weighted'),typ)
    print("F1 score (weighted) SVM Huber:  %f" % f1_score(y, y_pred_sgh_final, average='weighted'),typ)
    print("\n")
    print("Precision (micro) RF: %f" % precision_score(y, y_pred_rf_final, average='micro'),typ)
    print("Recall (micro) RF:    %f" % recall_score(y, y_pred_rf_final, average='micro'),typ)
    print("F1 score (micro) RF:  %f" % f1_score(y, y_pred_rf_final, average='micro'),typ, end='\n\n')
    print("Precision (macro) RF: %f" % precision_score(y, y_pred_rf_final, average='macro'),typ)
    print("Recall (macro) RF:    %f" % recall_score(y, y_pred_rf_final, average='macro'),typ)
    print("F1 score (macro) RF:  %f" % f1_score(y, y_pred_rf_final, average='macro'),typ, end='\n\n')
    print("Precision (weighted) RF: %f" % precision_score(y, y_pred_rf_final, average='weighted'),typ)
    print("Recall (weighted) RF:    %f" % recall_score(y, y_pred_rf_final, average='weighted'),typ)
    print("F1 score (weighted) RF:  %f" % f1_score(y, y_pred_rf_final, average='weighted'),typ)
    return [sgd, sgd_huber, rf]


def test_classifiers(test_vecs, test_labels, classifiers, typ='bow'):
    X = np.array(test_vecs)
    y = np.array(test_labels)
    scores = []
    for j,c in enumerate(classifiers):
        y_pred = classifiers[j].predict(X)
        scores.append(f1_score(y, y_pred, average='weighted'))
        print("Score for classifier %s %d: %f"%(typ,j,scores[-1]))
        
    return scores
        
    

#def match_topic_label(model, corpus, labels, label_list, number_of_classes=10):
#    count = [None] * number_of_classes
#    label_avg_score = [None] * number_of_classes
#    label_avg_score_t = [None] * number_of_classes
#    for i,c in enumerate(count):
#        count[i] = [0] * number_of_classes
#        label_avg_score[i] = [0] * number_of_classes
#        label_avg_score_t[i] = [0] * number_of_classes
#    scores = []
#    topics = []
#    for i,f in enumerate(corpus):
#        label_index = label_list.index(labels[i])
#        topic = model[corpus[i]]
#        if count[label_index][topic[0][0]]: count[label_index][topic[0][0]] += 1
#        else: count[label_index][topic[0][0]] = 1
#        scores.append(topic[0][1])
#        label_avg_score[label_index][topic[0][0]] = label_avg_score[label_index][topic[0][0]] + topic[0][1]
##        print(labels[i],scores[-1],label_avg_score[label_index][topic[0][0]])
#        label_avg_score_t[label_index][topic[0][0]] = label_avg_score_t[label_index][topic[0][0]] + topic[0][1]
#        topics.append(topic[0][0])
#        
#    for i,l in enumerate(label_list):
#        for j in range(number_of_classes):
#            if not count[i][j] == 0:
#                label_avg_score[i][j] = label_avg_score[i][j] / count[i][j]
##    for i,l in enumerate(label_list):
##        print(label_list[i],label_avg_score[i],label_avg_score_t[i],count[i],"\n")
##    import sys
##    sys.exit(1)
#        
#    tps = model.print_topics(-1)
#    for b,c in enumerate(count):
#        max_data = sorted(count[b], reverse=True)[:3]
##        m = max(count[b])
#        max_indexes = [count[b].index(j) for j in max_data]
#        if len(max_data) < 5:
#            for i,ds in enumerate(max_data):
#                print(str(b+1)+"): Label "+label_list[b]+ " corresponds with topic "+str(max_indexes[i]))
#                print("Topic: "+tps[max_indexes[i]][1])
#        else:
#            print(str(b+1)+"): Label "+label_list[b]+ " has no clear topic match")
#        
##        for index, score in sorted(model[bow_corpus[i]], key=lambda tup: -1*tup[1]):
##            print(index, score, labels[i])
##            print(model.get_document_topics(bow_corpus[i]))
##            print("\nScore: {}\t \nTopic: {}".format(score, model.print_topic(index, 10)))
#    return scores, topics
        
    
if __name__ == "__main__":
    
    print("Loading and splitting data for training and testing.\n")
    data = dp.load_data()
    training_data, testing_data = dp.get_split_data(data)
    files = list(data.keys())
    training_files = []
    training_labels = []
    training_content = []
    testing_files = []
    testing_labels = []
    testing_content = []
    training_keys = list(training_data.keys())
    for i,f in enumerate(files):
        if files[i] in training_keys:
            training_files.append(files[i])
            training_labels.append(training_data[files[i]]['label'])
            training_content.append(training_data[files[i]]['content'])
        else:
            testing_files.append(files[i])
            testing_labels.append(testing_data[files[i]]['label'])
            testing_content.append(testing_data[files[i]]['content'])
            
    print("Finished loading and splitting data for training and testing.\n")

    print("Preprocessing data for training.\n")
    #removing stop words and lemmatizing
    preprocessed_content_training = []
    #Bigrams and trigrams
    bigrams_content_training = []
    trigrams_content_training = []
    for i,t in enumerate(training_content):
        words = []
        for word in training_content[i].split(' '):
            words.append(word)
        preprocessed_content_training.append(dp.preprocess(training_content[i]))
    print("Finished preprocessing data for training.\n")

    #preprocess for bigrams and trigrams
    print("Getting bigram for training.\n")
    bigrams_content_training, bigram_mod = dp.get_bigrams(preprocessed_content_training)
    print("Finished getting bigram for training.\n")
    print("Getting trigram for training.\n")
    trigrams_content_training, trigram_mod = dp.get_trigrams(bigrams_content_training)
    print("Finished getting trigram for training.\n")
        
    print("Getting bow corpus and dictionary for training.\n")
    bow_corpus_training, dictionary = dp.get_dictionary_corpus(preprocessed_content_training)
    print("Finished getting bow corpus and dictionary for training.\n")
    
#    vectorizer = CountVectorizer()
#    X = vectorizer.fit_transform(bow_corpus_training)
#    print(X.toarray())
    
    #prepare data for bigram and trigram models
    print("Getting bigram corpus and dictionary for training.\n")
    bi_corpus_training, bi_dictionary = dp.get_dictionary_corpus(bigrams_content_training, save_path_dict='extracted_data/lda_dictionary_bigram', save_path_bcorp='extracted_data/lda_bigram_corpus')
    print("Finished getting bigram corpus and dictionary for training.\n")
    print("Getting trigram corpus and dictionary for training.\n")
    tri_corpus_training, tri_dictionary = dp.get_dictionary_corpus(trigrams_content_training, save_path_dict='extracted_data/lda_dictionary_trigram', save_path_bcorp='extracted_data/lda_trigram_corpus')
    print("Finished getting trigram corpus and dictionary for training.\n")
    
    l = list(set(training_labels + testing_labels))
    
    #Do we want to use this as tentative number of topics? Appears not to be a senseible solution. 
    #Better use HDP according to https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28
    number_of_classes = len(l)
    #HDP has a bug with python 3.8
#    hdp = models.HdpModel(bow_corpus, dictionary)
#    print(hdp.print_topics())

    #LDA with BOW
    print("Training bow LDA model.\n")
    lda_model_bow = lda_model(bow_corpus_training, dictionary)
    print("Finished training bow LDA model.\n")
    #LDA with bigrams and trigrams
    print("Training bigram LDA model.\n")
    lda_model_bi = lda_model(bi_corpus_training, bi_dictionary, number_of_topics=20, save_path='saved_models/lda_bi')
    print("Finished training bigram LDA model.\n")
    print("Training trigram LDA model.\n")
    lda_model_tri = lda_model(tri_corpus_training, tri_dictionary, number_of_topics=20, save_path='saved_models/lda_tri')
    print("Finished training trigram LDA model.\n")
    
    #Transform topics into features
    print("Getting training topic features for bow model.\n")
    train_vecs = get_topics_features(lda_model_bow, bow_corpus_training)
    print("Finsihed getting training topic features for bow model.\n")
    #Bigrams and trigrams
    print("Getting training topic features for bigram model.\n")
    train_vecs_bi = get_topics_features(lda_model_bi, bi_corpus_training)
    print("Finsihed getting training topic features for bigram model.\n")
    print("Getting training topic features for trigram model.\n")
    train_vecs_tri = get_topics_features(lda_model_tri, tri_corpus_training)
    print("Finsihed getting training topic features for trigram model.\n")
    
    #Train some classifiers
    print("Training bow classifiers.\n")
    classifiers = train_classifiers(train_vecs, training_labels)
    print("Finished training bow classifiers.\n")
    #Bigrams and trigrams
    print("Training bigram classifiers.\n")
    classifiers_bi = train_classifiers(train_vecs_bi, training_labels, typ='bi')
    print("Finished training bigram classifiers.\n")
    print("Training trigram classifiers.\n")
    classifiers_tri = train_classifiers(train_vecs_tri, training_labels, typ='tri')
    print("Finished training trigram classifiers.\n")
    
    print("Preprocessing data for testing.\n")
    #removing stop words and lemmatizing for testing data
    preprocessed_content_testing = []
    #Bigrams and trigrams
    bigrams_content_testing = []
    trigrams_content_testing = []
    for i,t in enumerate(testing_content):
        words = []
        for word in testing_content[i].split(' '):
            words.append(word)
        preprocessed_content_testing.append(dp.preprocess(testing_content[i]))
    print("Finished preprocessing data for testing.\n")
        
    #preprocess for bigrams and trigrams
    print("Getting bigram for testing.\n")
    bigrams_content_testing, __ = dp.get_bigrams(preprocessed_content_testing, bigram_mod, False)
    print("Finished getting bigram for testing.\n")
    print("Getting trigram for testing.\n")
    trigrams_content_testing, __ = dp.get_trigrams(bigrams_content_testing, trigram_mod, False)
    print("Finished getting trigram for testing.\n")
    
    #Get corpus for testing
    print("Getting bow corpus for testing.\n")
    bow_corpus_testing = dp.get_corpus(preprocessed_content_testing)
    print("Finished getting bow corpus for testing.\n")
    #Bigrams and trigrams
    print("Getting bigram corpus for testing.\n")
    bi_corpus_testing = dp.get_corpus(bigrams_content_testing)
    print("Finished getting bigram corpus for testing.\n")
    print("Getting trigram corpus for testing.\n")
    tri_corpus_testing = dp.get_corpus(trigrams_content_testing)
    print("Finished getting trigram corpus for testing.\n")
    
    #Transform topics into features for testing data
    print("Getting testing topic features for bow model.\n")
    test_vecs = get_topics_features(lda_model_bow, bow_corpus_testing)
    print("Finsihed getting testing topic features for bow model.\n")
    #Bigrams and trigrams
    print("Getting testing topic features for bigram model.\n")
    test_vecs_bi = get_topics_features(lda_model_bi, bi_corpus_testing)
    print("Finsihed getting testing topic features for bigram model.\n")
    print("Getting testing topic features for trigram model.\n")
    test_vecs_tri = get_topics_features(lda_model_tri, tri_corpus_testing)
    print("Finsihed getting testing topic features for bigram model.\n")
    
    #Test some classifiers
    print("Getting classifiers scores for bow model.\n")
    scores = test_classifiers(test_vecs, testing_labels, classifiers, typ='bow')
    print("Finshed getting classifiers scores for bow model.\n")
    #Bigrams and trigrams
    print("Getting classifiers scores for bigram model.\n")
    scores_bi = test_classifiers(test_vecs_bi, testing_labels, classifiers_bi, typ='bi')
    print("Finshed getting classifiers scores for bigram model.\n")
    print("Getting classifiers scores for trigram model.\n")
    scores_tri = test_classifiers(test_vecs_tri, testing_labels, classifiers_tri, typ='tri')
    print("Finshed getting classifiers scores for trigram model.\n")
    
#    print("Start tfidf model training and testing")
#    #Takes forever
#    tfidf_corpus_training = dp.get_tfidf(bow_corpus_training)
##    #LDA with TF-IDF
#    lda_model_tfidf = lda_model_tfidf(tfidf_corpus_training, dictionary)
#    train_vecs_tfidf = get_topics_features(lda_model_tfidf, tfidf_corpus_training)
#    classifiers_tfidf = train_classifiers(train_vecs_tfidf, training_labels)
#    
#    tfidf_corpus_testing = dp.get_tfidf(bow_corpus_testing)
#    test_vecs_tfidf = get_topics_features(lda_model_tfidf, tfidf_corpus_testing)
#    scores_tfidf = test_classifiers(test_vecs_tfidf, testing_labels, classifiers, typ='tfidf')
#    print("Finished tfidf model training and testing")

    
#    scores_tfidf, topics_tfidf = match_topic_label(lda_model_tfidf, tfidf_corpus, training_labels, l, number_of_classes)
    
    #Custom evaluation. Might not make sense. Preferably to use combination of topics as features
#    scores_bow, topics_bow = match_topic_label(lda_model_bow, bow_corpus, training_labels, l, number_of_classes)
#    x = [randint(0, len(training_files)) for p in range(0, 19)]
#    for i in x:
#        print("File "+training_files[i]+" with label "+training_labels[i]+" corresponds to bow topic "\
#              +str(topics_bow[i])+" with score "+str(scores_bow[i]))
#        print("File "+training_files[i]+" with label "+training_labels[i]+" corresponds to tfidf topic "\
#              +str(topics_bow[i])+" with score "+str(scores_bow[i]))

        


    