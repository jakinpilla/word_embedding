# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:31:43 2018

@author: dsc
"""

import numpy as np
import pandas as pd
from konlpy.utils import pprint
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
classifier.show_most_informative_features()

# sentiment classification with doc2vec
from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]


from gensim.models import doc2vec

# build dic
model= doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, seed=1234)
model.build_vocab(tagged_train_docs)

vector_size=300
window_size = 15
word_min_count = 2
sampling_threshold =1e-5
negative_size = 5
train_epoch = 100
dm = 1

# train document vectors
model.train(tagged_train_docs, epochs=10, total_examples=model.corpus_count)

pprint(model.wv.most_similar('북한/Noun'))
pprint(model.wv.most_similar('최순실/Noun'))
pprint(model.wv.most_similar('박근혜/Noun'))

# making features
X_train = [model.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags[0] for doc in tagged_train_docs]
X_test = [model.infer_vector(doc.words) for doc in tagged_test_docs]
y_test = [doc.tags[0] for doc in tagged_test_docs]

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import NearestCentroid

lr = LogisticRegression() 
svc = SVC(kernel="linear")
tree = DecisionTreeClassifier()
mlp = MLPClassifier()
ridge  =RidgeClassifier(tol=1e-2, solver='lsqr', alpha=.5)
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, 
                    max_iter=100, tol=None)
rf = RandomForestClassifier(max_features=9, n_estimators=100)
percep = Perceptron(n_iter=50)
pass_agg = PassiveAggressiveClassifier(n_iter=50)
near_cent = NearestCentroid()

from sklearn.metrics import classification_report
for clf in (lr, svc, tree, mlp, ridge, sgd, rf, percep, pass_agg, near_cent):
    clf.fit(X_train, y_train)
    print('=' * 25 + "    " + clf.__class__.__name__ + "    " + "="*30)
    print(clf.__class__.__name__, classification_report(y_test, clf.predict(X_test)));


