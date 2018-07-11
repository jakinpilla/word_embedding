# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:48:39 2018

@author: dsc
"""

import requests
import lxml.html
import datetime

import pandas as pd
import numpy as np

from lxml.html.clean import Cleaner
cleaner = Cleaner()
cleaner.javascript = True 
cleaner.style = True 

# datetime_list
datetime_list = []
start = datetime.date(2017, 1, 1)
end = datetime.date(2017, 12, 31)
days=[start + datetime.timedelta(days=x) for x in range((end-start).days + 1)]
for day in days:
    datetime_list.append(day.strftime('%Y%m%d'))
    
datetime_list[:5]

## Crawling

news_title_list = []
news_content_list = []

for datetime in datetime_list:
    print(datetime)
    url = "http://news.naver.com/main/ranking/popularDay.nhn?rankingType=popular_day&date=" + datetime
    res = requests.get(url)
    element = lxml.html.fromstring(res.text)
    
    # url of  first news of naver ranking news 
    postings = element.cssselect('div.ranking_section li.num1 a')
    news_url = postings[0].attrib["href"]
    
    # bring news text from the urls
    res_2 = requests.get("http://news.naver.com" + news_url)
    element_2 = lxml.html.fromstring(res_2.text)
    
    # news_title
    news_title = element_2.cssselect("div.article_info h3")[0].text_content()
    
    # news_content
    news_content = cleaner.clean_html(element_2.get_element_by_id("articleBodyContents")).text_content()
    
    # Data PrepProcessing 
    # remove unnecessary characters #
    while news_content.find("\n") != -1 :
        news_content = news_content.replace("\n", " ")
        
    while news_content.find("    ") != -1 :
        news_content = news_content.replace("    ", " ")
               
    # remove string from email   
    if news_content.find("@") !=-1:
        index = news_content.find("@")
        while news_content[index] not in [" ", "."]:
            index = index -1
        news_content = news_content[:index]      

    news_content = news_content.strip()

    ########################################
    
    news_title_list.append(news_title)
    news_content_list.append(news_content)


# news dataframe
df_news = pd.DataFrame(columns=["Date", "Title", "Content"])
df_news["Date"] = datetime_list
df_news["Title"] = news_title_list
df_news["Content"] = news_content_list    
df_news.head()  

# load news_dataset labels
# find and upload label data .csv file

    