# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:19:45 2018

@author: dsc
"""

import numpy as np
import pandas as pd

news_data = pd.read_csv('./data/news_dataset.csv', index_col=0)
news_data.columns
news_data['label'].value_counts().plot(kind = 'bar')

from konlpy.tag import Twitter
from konlpy.utils import pprint
pos_tagger = Twitter()

#def kor_noun(text):
#    words = []
#    tagged = tag.nouns(text)
#    for i in range(0, len(tagged)):
#        if len(tagged[i]) > 1:
#            words.append(tagged[i])
#    return words

pos_tagger.pos("안녕하세요? 김정규입니다. 저는 파이썬 개발자입니다.", norm=True, stem=True)

for i in pos_tagger.pos("안녕하세요? 김정규입니다. 저는 파이썬 개발자입니다.", norm=True, stem=True):
    if ((i[1] == 'Noun') | (i[1] == 'Adjective')) & (len(i[0]) > 1):
        print(i[0])

def tokenize(doc):
    words = []
    for i in pos_tagger.pos(doc, norm=True, stem=True):
        if ((i[1] == 'Noun') | (i[1] == 'Adjective')) & (len(i[0]) > 1):
            tagged_word = i[0] + '/' + i[1]
            words.append(tagged_word)
    return [t for t in words if t not in ['이다/Adjective', '기자/Noun']]

tokenize("안녕하세요? 김정규입니다. 저는 파이썬 개발자입니다.")
news_data['content'][0]
tokenize(news_data['content'][0])

words_per_content = []    
for i in range(len(news_data)):
    words_per_content.append(tokenize(news_data['content'][i]))
    
words_per_content[0]
words_per_content[1]
len(words_per_content)

word_df = pd.DataFrame({'words' : words_per_content})
word_df.head()

words_concat = []
for i in range(len(word_df)):
    words_concat.append(' '.join(word_df['words'][i]))
    
word_concat_df = pd.DataFrame({'words_concat' : words_concat})
word_concat_df.head()

len(words_per_content)
words_per_content[:5]
# save sentence_list(=words_per_content) in a file with pickle
import pickle
with open('./data/sentence_list.txt', 'wb') as fp:
    pickle.dump(words_per_content, fp)

with open('./data/sentence_list.txt', 'rb') as fp:
    sentence_list = pickle.load(fp)
    
sentence_list[:2]
len(sentence_list)

news_data.head()
y = np.array(news_data['label'])
label_list=y.tolist()

data_df = pd.DataFrame({'sentence_list' : sentence_list, 
                        'label' : label_list})
data_df.head()
data_df.to_csv('./data/processed_data.csv')
pd.read_csv('./data/processed_data.csv', index_col=0).head()

from sklearn.model_selection import train_test_split
X = sentence_list
X[:1]
y = label_list
y[:1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train[:1]
len(X_train)
y_train[:1]
len(y_train)
X_test[:1]
len(X_test)
y_test[:1]
len(y_test)

X_train[0]
y_train[0]

# train_docs
train_docs = [(X_train[i], y_train[i]) for i in range(len(X_train))]
train_docs[0]
train_docs[1]

import pickle
with open('./data/train_docs.txt', 'wb') as fp:
    pickle.dump(train_docs, fp)

with open('./data/train_docs.txt', 'rb') as fp:
    train_docs_loaded = pickle.load(fp)

train_docs[0] ; len(train_docs[0][0])
train_docs_loaded[0] ; len(train_docs_loaded[0][0])

# test_docs
test_docs = [(X_test[i], y_test[i]) for i in range(len(X_test))]
test_docs[0]
test_docs[1]

with open('./data/test_docs.txt', 'wb') as fp:
    pickle.dump(test_docs, fp)

with open('./data/test_docs.txt', 'rb') as fp:
    test_docs_loaded = pickle.load(fp)

test_docs[0] ; len(test_docs[0][0])
test_docs_loaded[0] ; len(test_docs_loaded[0][0])

# how many tokens...
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

import nltk
text = nltk.Text(tokens)
print(text)

print(len(text.tokens))
print(len(set(text.tokens)))

pprint(text.vocab().most_common(10))

# text.plot()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
f = [f.name for f in fm.fontManager.ttflist]
[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Malgun' in f.name]
plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(15, 4))
text.plot(50)

























