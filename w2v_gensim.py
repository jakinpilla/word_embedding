# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:36:19 2018

@author: dsc
"""
import gensim
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import pickle
from EmbeddingVectorizer import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer

with open('./data/sentence_list.txt', 'rb') as fp:
    sentence_list = pickle.load(fp)

sentence_list[:1]
len(sentence_list)

model= gensim.models.word2vec.Word2Vec(sentences=sentence_list, sg=0, 
                                       min_count=5, size=100, window=5)

# print(model)
# print(model.wv.vocab.keys())

model.most_similar(positive=['최순실/Noun'], topn=30)
model.most_similar(positive=['북한/Noun'], topn=30)
model.most_similar(positive=['정당/Noun'], topn=30)
model.most_similar(positive=['박근혜/Noun'], topn=30)
model.most_similar(positive=['탄핵/Noun'], topn=30)

model['국방/Noun']

w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
w2v

# model definition
mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), 
                    ("multinomial nb", MultinomialNB())])
    
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), 
                    ("bernoulli nb", BernoulliNB())])
    
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), 
                          ("multinomial nb", MultinomialNB())])
    
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), 
                          ("bernoulli nb", BernoulliNB())])
    
# SVM - which is supposed to be more or less state of the art 
# http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), 
                ("linear svc", SVC(kernel="linear"))])
    
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
                      ("linear svc", SVC(kernel="linear"))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])

    
# prepare data
X = np.array(sentence_list)
X
X.shape
news_dataset = pd.read_csv('./data/news_dataset.csv', index_col=0)
news_dataset.columns
y = np.array(news_dataset['label'])
y


all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("bern_nb", bern_nb),
    ("bern_nb_tfidf", bern_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("w2v", etree_w2v),
    ("w2v_tfidf", etree_w2v_tfidf),
]

# scores
scores = sorted([(name, cross_val_score(model, X, y, cv=5).mean()) 
                 for name, model in all_models])

print(tabulate(scores, headers=('model', 'score')))

def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))
    scores = []
    for train, test in StratifiedShuffleSplit(y, n_iter=5, test_size=test_size):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)

train_sizes = [10, 40, 160, 350]
table = []
for name, model in all_models:
    for n in train_sizes:
        table.append({'model': name, 
                      'accuracy': benchmark(model, X, y, n), 
                      'train_size': n})
    
df = pd.DataFrame(table)
df

plt.figure(figsize=(12, 6))
fig = sns.pointplot(x='train_size', y='accuracy', hue='model', 
                    data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "w2v_tfidf"])])
sns.set_context("notebook", font_scale=1.5)
fig.set(ylabel="accuracy")
fig.set(xlabel="labeled training examples")
fig.set(title="Benchmark Reult")
fig.set(ylabel="accuracy")



