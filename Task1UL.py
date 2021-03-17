#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:53:27 2020

@author: tobias
"""


#!/usr/bin/env python3

import pprint # pretty printer
import logging
from sklearn.datasets import fetch_20newsgroups
from fda_helper import preprocess_data
from gensim import corpora, models, similarities

# enable logging to display what is happening
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# read dataset 20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

texts = preprocess_data(documents)
dictionary = corpora.Dictionary(texts)

bow_corpus = [dictionary.doc2bow(text) for text in texts] # bow = Bag Of Words
#pprint.pprint(bow_corpus[5]) # one example document, words maped to ids

tfidf = models.TfidfModel(bow_corpus) # train tf-idf model
corpus_tfidf = tfidf[bow_corpus] # apply transformation on the whole corpus

#transform your tfidf model into a LSI Model
#using python gensim, use num_topics=200

lsi_model = models.LsiModel(corpus_tfidf, num_topics=200, id2word=dictionary)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi_model.print_topics(200)

#pick random document
doc = documents[502]
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi_model[vec_bow]  # convert the query to LSI space

#initialize a query structure for your LSI space
index = similarities.MatrixSimilarity(lsi_model[corpus_tfidf])  # transform corpus to LSI space and index it

#perform the query on the LSI space, interpret the result and summarize your findings in the report

res = index[vec_lsi]  # perform a similarity query against the corpus
res = sorted(enumerate(res), key=lambda item: -item[1]) #sort these similarities into descending order
print(res)