#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:12:27 2020

@author: odrec
"""
import gensim
from gensim import models

from os.path import isfile
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')

import get_data_classes as gdc

def load_data():
    data = gdc.load_extracted_data()
    return data

def get_split_data(data, split=0.8):
    training_data, testing_data = gdc.get_train_test_split(data, split)
    return training_data, testing_data

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
#        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
        if token not in gensim.parsing.preprocessing.STOPWORDS:
#            result.append(lemmatize_stemming(token))
            result.append(token)
    return result

def get_bigrams(words, bigram_mod=None, train=True):
    if train:
        bigrams = gensim.models.Phrases(words, min_count=1, threshold=1)
        bigram_mod = gensim.models.phrases.Phraser(bigrams)
    
    bigram_words = []
    for s in words:
        bigram_words.append(bigram_mod[s])

    return bigram_words, bigram_mod

def get_trigrams(bigram_words, trigram_mod=None, train=True):
    if train:
        trigrams = gensim.models.Phrases(bigram_words, min_count=1, threshold=1)
        trigram_mod = gensim.models.phrases.Phraser(trigrams)
    
    trigram_words = []
    for s in bigram_words:
        trigram_words.append(trigram_mod[s])
        
    return trigram_words, trigram_mod

def get_dictionary_corpus(data, save_path_dict='extracted_data/lda_dictionary', save_path_bcorp='extracted_data/lda_bow_corpus'):
    if isfile(save_path_dict):
        dictionary = Dictionary.load_from_text(save_path_dict)
        corpus = gensim.corpora.MmCorpus(save_path_bcorp)
    else:
        dictionary = gensim.corpora.Dictionary(data)
        dictionary.filter_extremes(no_above=0.5, keep_n=100000)
        corpus = [dictionary.doc2bow(doc) for doc in data]
        dictionary.save_as_text(save_path_dict)
        gensim.corpora.MmCorpus.serialize(save_path_bcorp, corpus)
    
    #    bow_doc_2 = bow_corpus[2]
    #    for i in range(len(bow_doc_2)):
    #        print("Word {} (\"{}\") appears {} time.".format(bow_doc_2[i][0], dictionary[bow_doc_2[i][0]], bow_doc_2[i][1]))
        
    return corpus, dictionary

def get_corpus(data, save_path_dict='extracted_data/lda_dictionary'):
    if isfile(save_path_dict):
        dictionary = Dictionary.load_from_text(save_path_dict)
        corpus = [dictionary.doc2bow(doc) for doc in data]
        return corpus
    else:
        print("Didn't find a dictionary.")
        import sys
        sys.exit(1)
    
def get_tfidf(bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf