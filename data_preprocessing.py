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

from konlpy.tag import Komoran
tag = Komoran()

def kor_noun(text):
    words = []
    tagged = tag.nouns(text)
    for i in range(0, len(tagged)):
        if len(tagged[i]) > 1:
            words.append(tagged[i])
    return words

kor_noun('안녕하세요? 김정규입니다. 저는 파이썬 개발자입니다.')

words_per_content = []    
for i in range(len(news_data)):
    words_per_content.append(kor_noun(news_data['content'][i]))
    
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
