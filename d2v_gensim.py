# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:31:43 2018

@author: dsc
"""

import numpy as np
import pandas as pd

import pickle

# loading trian_docs 
with open('./data/train_docs.txt', 'rb') as fp:
    train_docs = pickle.load(fp)

train_docs[0] ; len(train_docs[0][0])

# loading test_docs
with open('./data/test_docs.txt', 'rb') as fp:
    test_docs = pickle.load(fp)

test_docs[0] ; len(test_docs[0][0])

# define tokens and text
tokens = [t for d in train_docs for t in d[0]]
import nltk
text = nltk.Text(tokens)

# sentiment classification with term existance
selected_words = [f[0] for f in text.vocab().most_common(2000)]
selected_words[:5]

def term_exists(doc):
    return {'exists({})'.format(word): (word in set(doc)) for word in selected_words}
    
train_xy = [(term_exists(d), c) for d, c in train_docs]
train_xy[:5]
test_xy = [(term_exists(d), c) for d, c in test_docs]
train_xy[:1]

classifier = nltk.NaiveBayesClassifier.train(train_xy)
print(nltk.classify.accuracy(classifier, test_xy))
classifier.show_most_informative_features(10)

# sentiment classification with doc2vec
from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'word tags')
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs =[TaggedDocument(d, [c]) for d, c in test_docs]

tagged_train_docs

from gensim.models import doc2vec


