import json
import os
import time
import datetime
import pandas as pd
import numpy as np
import regex as re
import sqlalchemy
import config
import helper_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from py3pin.Pinterest import Pinterest
from instaloader import Instaloader, Hashtag


# Pinterest crawler class
class PinterestCrawler(Pinterest):
    '''
    Based on https://github.com/bstoilov/py3-pinterest install using:
    > pip install py3-pinterest
    '''
    def __init__(self, username, password):
        super().__init__(password=password, proxies=None, username='', email=username, cred_root='data', user_agent=None)
        # self.home_page = 'https://gr.pinterest.com/login/?referrer=home_page'        
        self.home_page = 'https://gr.pinterest.com/'        
        self.fashion_att = helper_functions.get_fashion_attributes()

    def _search(self, query, threshold):
        engine = helper_functions.ENGINE

        # productsDF = pd.read_sql_query('''SELECT * FROM S4F.dbo.Product''', engine)
        productsDF = pd.read_sql_query('''SELECT * FROM  public.\"Product\"''', engine)

        results = self.search('pins', query, page_size=threshold)
        query_result = []
        for index, result in enumerate(results, start=1):
            if result['type'] == 'pin' and result['videos'] == None:
                tempDF = productsDF.loc[productsDF['URL'] == result['link']]
                promotion = result['is_promoted']
                if tempDF.empty and not promotion:
                    searchwords = ''.join(query.split(" "))
                    imageFilePath = helper_functions.setImageFilePath(self.home_page, searchwords, index)
                    # Capture the main attributes of the result to evaluate and enchance the final recommendation
                    query_result.append({'query':query,
                                    'timestamp': str(datetime.datetime.now()),
                                    'URL': result['link'],
                                    'imgURL':result['images']['orig']['url'], 
                                    'imageFilePath':imageFilePath, 
                                    'title':result['title'], 
                                    'description':result['description']})
                else:
                    print('Product already exists')
        return query_result


    def search_query(self, query, threshold=10):
        # Initial search, may return existing products
        query_result = self._search(query, threshold)
        # Setup new search constrains
        _thresh = threshold-len(query_result)
        while len(query_result)<threshold:     
            _q = self._search(query, _thresh)
            query_result += _q
            _thresh = threshold-len(query_result)
        
        return query_result[:threshold]
        

# Instagram crawler class
class InstagramCrawler():
    '''
    Based on https://instaloader.github.io/ install using:
    > pip install instaloader
    '''
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.instagram = Instaloader()
        self.home_page = 'https://www.instagram.com'        
        self.base_url = 'https://www.instagram.com/p/'        
        self.fashion_att = helper_functions.get_fashion_attributes()

    def login(self,):
        self.instagram.login(self.username, self.password)

    def search_query(self, query, threshold=10):
        engine = helper_functions.ENGINE

        # productsDF = pd.read_sql_query('''SELECT * FROM S4F.dbo.Product''', engine)
        productsDF = pd.read_sql_query('''SELECT * FROM  public.\"Product\"''', engine)

        # Create hashtag from query term
        hashtag = Hashtag.from_name(self.instagram.context, query.replace(' ',''))
        query_result = []
        n_exists = 0
        for index, result in enumerate(hashtag.get_posts(), start=1):
            if index - n_exists> threshold:
                break
            post_url = self.base_url + str(result.shortcode) + "/"
            imgSource = result.url
            tempDF = productsDF.loc[productsDF['URL'] == post_url]
            video = result.is_video
            imageFilePath = helper_functions.setImageFilePath(post_url, query.replace(' ',''), index)
            if tempDF.empty and not video:
                post_info = " ".join(re.findall("[a-zA-Z]+", result.caption))
                post_title = ' '.join(result.caption_hashtags)
                # post_hashtags = result.caption_hashtags
                # post_likes = result.likes
                post_date = result.date
                query_result.append(({'query': query,
                            'timestamp': post_date,
                            'URL': post_url,
                            'imgURL': imgSource,
                            'imageFilePath':imageFilePath,
                            'title': post_title,
                            'description': post_info}))
            else:
                print('Product already exists')
                n_exists += 1

            if index - n_exists> threshold:
                break
        return query_result

    
def save_ranked(query_result, login_page, segmentation=False):
    dataDF = pd.DataFrame(query_result)
    # Form metadata column
    dataDF['metadata'] = dataDF['title'].str.cat(dataDF['description'].astype(str), sep=' ')
    # Preprocess metadata
    dataDF['processed_metadata'] = dataDF['metadata'].apply(lambda row: helper_functions.preprocess_metadata(row, segmentation))
    # Preprocess query
    dataDF['query'] = dataDF['query'].apply(lambda row: ' '.join(helper_functions.preprocess_words(row.split())))

    ## Calculate a factor for tokens that appear in metatdata
    keywords = dataDF.iloc[0]['query'].split()
    dataDF['factor'] = dataDF['processed_metadata'].apply(lambda row: len(set([word for word in row.split() if word in keywords])) / len(keywords))

    ## Calculate a factor based on the cosine similarity of TFIDF transformation of the query terms and 
    # the processed metadata using the fashion expert terminology as vocabulary
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2)
    vectorizer.fit_transform(helper_functions.get_fashion_attributes())
    metadata_tfidf = vectorizer.transform(dataDF['processed_metadata'])
    query_tfidf = vectorizer.transform(dataDF['query'])

    ## Calculate cosine similarity
    cosine_vector = cosine_similarity(query_tfidf[0], metadata_tfidf)
    dataDF['cosine_ranking_score'] = np.hstack(cosine_vector).tolist() * dataDF['factor']

    ## Calculate a factor based on the social media recommendation algorithm (order of appearence)
    scaler = MinMaxScaler((0.5, 1.))
    social_media_score = scaler.fit_transform(np.arange(len(dataDF)).reshape(-1, 1))[::-1]
    dataDF.loc[dataDF.sort_values(by ='timestamp', ascending=False).index, 'social_media_score'] = social_media_score
    
    ## Calculate Final Ranking Score giving the cosine similarity factor a greater score than the 
    # factor based on the social media recommendation algorithm (order of appearence)
    dataDF['final_score'] = (dataDF['cosine_ranking_score'] * 0.7) + (dataDF['social_media_score'] * 0.3)
    dataDF.sort_values(by ='final_score', ascending=False, inplace=True)

    # Save ranked results to the database
    for _, row in dataDF.iterrows():
        site = login_page
        searchwords = query_result[0]['query']
        imageFilePath = row['imageFilePath']    
        url = row['URL']
        imgURL = row['imgURL']
        empPhoto = helper_functions.getImage(imgURL, imageFilePath)
        head = row['title']
        meta = row['description']
        helper_functions.addNewProduct(site, 
                                        searchwords, 
                                        imageFilePath, 
                                        empPhoto, 
                                        url,
                                        imgURL, 
                                        head, 
                                        None, 
                                        None, 
                                        None, 
                                        meta, 
                                        None, 
                                        None)
    return dataDF
