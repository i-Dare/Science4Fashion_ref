import json
import os
import time
import logging
import pandas as pd
import numpy as np
import regex as re
import sqlalchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from py3pin.Pinterest import Pinterest
from instaloader import Instaloader, Hashtag
from instaloader.exceptions import ConnectionException

from sqlite3 import OperationalError, connect
from glob import glob
from os.path import expanduser
from platform import system

import core.config as config
from core.helper_functions import *
from core.logger import S4F_Logger
from core.query_manager import QueryManager

# Pinterest crawler class
class PinterestCrawler(Pinterest):
    '''
    Based on https://github.com/bstoilov/py3-pinterest install using:
    > pip install py3-pinterest
    '''
    def __init__(self, crawlSearchID, searchTerm, threshold=10, user=config.DEFAULT_USER, 
            loglevel=config.DEFAULT_LOGGING_LEVEL):
        # Get input arguments
        self.crawlSearchID = crawlSearchID
        self.searchTerm = searchTerm
        self.threshold = int(threshold)
        self.user = user
        self.logging = S4F_Logger('PinterestLogger', user=user, level=loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)
        self.db_manager = QueryManager()

        # Retrieve credentials from configuration
        username = config.PINTEREST_USERNAME
        password = config.PINTEREST_PASSWORD

        cred_root = os.path.join(config.RESOURCESDIR, 'data')
        super().__init__(password=password, proxies=None, username='', email=username, cred_root=cred_root, user_agent=None)
        
        self.logger.info('Connected to Pinterest', {'CrawlSearch': self.crawlSearchID})
        self.home_page = 'https://gr.pinterest.com/'        
        self.fashion_att = self.helper.get_fashion_attributes()

    def _search(self, threshold):      
        results = self.search('pins', self.searchTerm, page_size=threshold)
        # Discard irrelevant results
        results = [r for r in results if r['type'] == 'pin' and r['videos'] == None]
        # Discard duplicates by accessing the DB Product table 
        existing_articles = self.db_manager.runSelectQuery(params={'table': 'Product', 'Adapter': 5}, 
                filters=['URL']).squeeze().tolist()
        results = [r for r in results if r['link'] not in existing_articles and r['link']]
        if len(results)>0:            
            query_result = []
            for index, result in enumerate(results, start=1):
                promotion = result['is_promoted']
                if not promotion:
                    searchwords = ''.join(self.searchTerm.split(" "))
                    imageFilePath = self.helper.setImageFilePath(self.home_page, searchwords, index)
                    description = result['description'] + result['rich_summary']['display_description'] \
                                        if result['rich_summary'] else result['description']
                    # Capture the main attributes of the result to evaluate and enchance the final recommendation
                    query_result.append({'searchTerm': self.searchTerm,
                                    'timestamp': str(datetime.now()),
                                    'URL': result['link'],
                                    'imgURL':result['images']['orig']['url'], 
                                    'imageFilePath':imageFilePath, 
                                    'title':result['title'], 
                                    'description': description })
                else:
                    pass
            return query_result
        else:
            self.logger.warning('No new items', {'CrawlSearch': self.crawlSearchID})
            return []

    def search_query(self,):
        max_threshold = 250
        if self.threshold > max_threshold:
            self.logger.warning('Number of requested items exceeds the maximum number of items, up to \
                    250 items will be collected.', {'CrawlSearch': self.crawlSearchID})
            self.threshold = 250
        # Execute query
        query_result = self._search(self.threshold)
        # Setup new search constrains
        breakCnt = 3 # Loop breaking constrain
        while len(query_result) < self.threshold:
            no_queries = len(query_result)
            query_result += self._search(self.threshold-len(query_result))
            if len(query_result) == no_queries:
                breakCnt -= 1
            else:
                breakCnt = 3
            if breakCnt == 0:
                break
        # Return 
        return query_result[:self.threshold]


    def executeCrawling(self,):
        # Login to Pinterest
        self.login()
        # Execute crawling in Pinterest 
        start_time_all = time.time()
        query_result = self.search_query()
        productIDs = []
        if len(query_result) > 0:
            # Store results in the database ranked by the relevance of the experts terminology
            productIDs = save_ranked(self.crawlSearchID, self.helper, query_result, 'Pinterest', self.home_page)

            self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(productIDs)),
                 {'CrawlSearch': self.crawlSearchID})
            self.logger.info("Time to scrape ALL queries is %s seconds ---" % 
                    round(time.time() - start_time_all, 2), {'CrawlSearch': self.crawlSearchID})
        else:
            self.logger.warning('No results in Pinterest for %s' % self.searchTerm, {'CrawlSearch': self.crawlSearchID})
        return productIDs

        

# Instagram crawler class
class InstagramCrawler():
    '''
    Based on https://instaloader.github.io/ install using:
    > pip install instaloader
    '''
    def __init__(self, crawlSearchID, searchTerm, threshold=10, user=config.DEFAULT_USER, 
            loglevel=config.DEFAULT_LOGGING_LEVEL):
        # Get input arguments
        self.crawlSearchID = crawlSearchID
        self.searchTerm = searchTerm
        self.threshold = int(threshold)
        self.user = user
        self.logging = S4F_Logger('InstagramLogger', user=user, level=loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)
        self.db_manager = QueryManager()

        # Retrieve credentials from configuration
        self.username = config.INSTAGRAM_USERNAME
        self.password = config.INSTAGRAM_PASSWORD
        self.instagram = Instaloader()

        self.home_page = 'https://www.instagram.com'        
        self.base_url = 'https://www.instagram.com/p/'        
        self.fashion_att = self.helper.get_fashion_attributes()

    def login(self,):
        try:
            self.instagram.login(self.username, self.password)
        except ConnectionException as ex:
            self.logger.warn_and_trace(ex)
            self.logger.warn('Login failed, attempting to login with Firefox session.', {'CrawlSearch': self.crawlSearchID})
            cookiefile = self.get_cookiefile()
            session_path = os.path.join(config.RESOURCESDIR, 'data', 'instagram_session')
            username = self.import_session(cookiefile, session_path)
            self.instagram.load_session_from_file(username, session_path)
        if self.instagram.context.is_logged_in:
            self.logger.info('Connected to Instagram', {'CrawlSearch': self.crawlSearchID})

    def get_cookiefile(self):
        default_cookiefile = {
            "Windows": "~/AppData/Roaming/Mozilla/Firefox/Profiles/*/cookies.sqlite",
            "Darwin": "~/Library/Application Support/Firefox/Profiles/*/cookies.sqlite",
        }.get(system(), "~/.mozilla/firefox/*/cookies.sqlite")
        cookiefiles = glob(expanduser(default_cookiefile))
        if not cookiefiles:
            raise SystemExit("No Firefox cookies.sqlite file found. Use -c COOKIEFILE.")
        return cookiefiles[0]

    def import_session(self, cookiefile, session_path):
        print("Using cookies from {}.".format(cookiefile))
        conn = connect(f"file:{cookiefile}?immutable=1", uri=True)
        try:
            cookie_data = conn.execute(
                "SELECT name, value FROM moz_cookies WHERE baseDomain='instagram.com'"
            )
        except OperationalError:
            cookie_data = conn.execute(
                "SELECT name, value FROM moz_cookies WHERE host LIKE '%instagram.com'"
            )
        _instagram = Instaloader(max_connection_attempts=3)
        _instagram.context._session.cookies.update(cookie_data)
        username = _instagram.test_login()
        if not username:
            raise SystemExit("Not logged in. \nLogin to Instagram through Firefox and try again")
        print("Imported session cookie for {}.".format(username))        
        _instagram.context.username = username
        _instagram.save_session_to_file(session_path)
        return username

    def search_query(self,):
        # Override due to unsolved #874: https://github.com/instaloader/instaloader/issues/874
        hashtag = self.searchTerm.replace(' ','')
        jsonData = self.instagram.context.get_json(path="explore/tags/" + hashtag + "/", params={"__a": 1})
        query_result = []
        n_exists = 0
        hasNextPage = True
        # Check for duplicates by accessing the DB Product table 
        existing_articles = self.db_manager.runSelectQuery(params={'table': 'Product', 'Adapter': 4}, 
                filters=['URL']).squeeze().tolist()

        while hasNextPage:
            sections = jsonData['data']['recent']['sections']
            for section in sections:
                for index, post in enumerate(section['layout_content']['medias']):
                    if post['media']['media_type'] != 1:
                        continue
                    post_url = self.base_url + post['media']['code']
                    imgSource = post['media']['image_versions2']['candidates'][0]['url']
                    meta = post['media']['caption']['text']
                    post_info = " ".join(re.findall("[a-zA-Z]+",meta))
                    imageFilePath = self.helper.setImageFilePath(post_url, self.searchTerm.replace(' ',''), index)
                    post_title = self.searchTerm
                    timestamp = datetime.fromtimestamp(post['media']['taken_at']).date()
                    if post_url in existing_articles:
                        query_result.append(({'searchTerm': self.searchTerm,
                                    'timestamp': timestamp,
                                    'URL': post_url,
                                    'imgURL': imgSource,
                                    'imageFilePath':imageFilePath,
                                    'title': post_title,
                                    'description': post_info}))
                    else:
                        self.logger.debug('Product already exists', {'CrawlSearch': self.crawlSearchID})
                        n_exists += 1
                    if len(query_result) == self.threshold:
                        return query_result
        # # Create hashtag from searchTerm term
        # hashtag = Hashtag.from_name(self.instagram.context, searchTerm.replace(' ',''))
        # query_result = []
        # n_exists = 0
        # for index, result in enumerate(hashtag.get_posts(), start=1):
        #     post_url = self.base_url + str(result.shortcode) + "/"
        #     imgSource = result.url
        #     isVideo = result.is_video
        #     imageFilePath = self.helper.setImageFilePath(post_url, searchTerm.replace(' ',''), index)
        #     filter_df = self.db_manager.runSelectQuery({'table': 'Product', 'URL': post_url})
        #     if type(filter_df)==pd.DataFrame:
        #         if filter_df.empty and not isVideo:
        #             post_info = " ".join(re.findall("[a-zA-Z]+", result.caption))
        #             post_title = ' '.join(result.caption_hashtags)
        #             # post_hashtags = result.caption_hashtags
        #             # post_likes = result.likes
        #             post_date = result.date
        #             query_result.append(({'searchTerm': searchTerm,
        #                         'timestamp': post_date,
        #                         'URL': post_url,
        #                         'imgURL': imgSource,
        #                         'imageFilePath':imageFilePath,
        #                         'title': post_title,
        #                         'description': post_info}))
        #         else:
        #             self.logger.info('Product already exists', {'CrawlSearch': self.crawlSearchID})
        #             n_exists += 1

        #     if index - n_exists> threshold:
        #         break
        return query_result
        

    def executeCrawling(self,):
        # Login to Instagram
        self.login()
        # Execute crawling in Instagram 
        start_time_all = time.time()
        query_result = self.search_query()
        productIDs = []
        if len(query_result) > 0:
            # Store results in the database ranked by the relevance of the experts terminology
            productIDs = save_ranked(self.crawlSearchID, self.helper, query_result, 'Instagram', self.home_page)

            self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(productIDs)),
                 {'CrawlSearch': self.crawlSearchID})
            self.logger.info("Time to scrape ALL queries is %s seconds ---" % 
                    round(time.time() - start_time_all, 2), {'CrawlSearch': self.crawlSearchID})
        else:
            self.logger.warning('No results in Instagram for %s' % self.searchTerm, {'CrawlSearch': self.crawlSearchID})
        return productIDs

    
def save_ranked(crawlSearchID, helper, query_result, adapter, standardUrl, segmentation=True):
    dataDF = pd.DataFrame(query_result)
    # Dropping duplicates
    dataDF = dataDF.loc[dataDF['imgURL'].drop_duplicates().index].reset_index(drop=True)
    # Form metadata column
    dataDF['metadata'] = dataDF['title'].str.cat(dataDF['description'].astype(str), sep=' ')
    dataDF.loc[dataDF['metadata'].isnull(), 'metadata'] = ''
    # Preprocess metadata
    dataDF['processed_metadata'] = dataDF['metadata'].apply(lambda row: helper.preprocess_metadata(row, segmentation))
    # Preprocess searchTerm
    dataDF['searchTerm'] = dataDF['searchTerm'].apply(lambda row: ' '.join(helper.preprocess_words(row.split())))

    ## Calculate a factor for tokens that appear in metatdata
    keywords = dataDF.iloc[0]['searchTerm'].split()
    dataDF['factor'] = dataDF['processed_metadata'].apply(lambda row: \
            len(set([word for word in row.split() if word in keywords])) / len(keywords))

    ## Calculate a factor based on the cosine similarity of TFIDF transformation of the searchTerm terms and 
    # the processed metadata using the fashion expert terminology as vocabulary
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2)
    vectorizer.fit_transform(helper.get_fashion_attributes())
    metadata_tfidf = vectorizer.transform(dataDF['processed_metadata'])
    query_tfidf = vectorizer.transform(dataDF['searchTerm'])

    ## Calculate cosine similarity of searchTerm tokens and metadata tokens
    cosine_vector = cosine_similarity(metadata_tfidf, query_tfidf).sum(axis=1)
    dataDF['cosine_ranking_score'] = np.hstack(cosine_vector).tolist() * dataDF['factor']

    ## Calculate a factor based on the social media recommendation algorithm (order of appearence)
    scaler = MinMaxScaler((0.5, 1.))
    social_media_score = scaler.fit_transform(np.arange(len(dataDF)).reshape(-1, 1))[::-1]
    dataDF.loc[dataDF.sort_values(by ='timestamp', ascending=False).index, 'social_media_score'] = social_media_score
    
    ## Calculate Final Ranking Score giving the cosine similarity factor a greater score than the 
    # factor based on the social media recommendation algorithm (order of appearence)
    dataDF['final_score'] = (dataDF['cosine_ranking_score'] * 0.7) + (dataDF['social_media_score'] * 0.3)
    dataDF = dataDF.sort_values(by ='final_score', ascending=False).reset_index(drop=True)
    dataDF['ReferenceOrder'] = dataDF.index
    dataDF['TrendingOrder'] = dataDF.reset_index(drop=True).sort_values(by ='social_media_score', ascending=False).index

    # Save ranked results to the database
    productIDs = []
    for _, row in dataDF.iterrows():
        site = adapter
        searchTerm = query_result[0]['searchTerm']
        imageFilePath = row['imageFilePath']    
        url = row['URL']
        imgURL = row['imgURL']
        empPhoto = helper.saveImage(imgURL, imageFilePath)
        head = row['title']
        referenceOrder = row['ReferenceOrder']
        trendOrder = row['TrendingOrder']
        meta = row['processed_metadata']
        uniq_params = {'table': 'Product', 'URL': url}
        params = {'table': 'Product', 'Description': searchTerm, 'Active':  True, 'Ordering': 0, 
                 'ProductTitle': head, 'SiteHeadline': head, 'Metadata': meta, 'URL': url, 
                 'ImageSource': imgURL, 'Image': empPhoto, 'Photo': imageFilePath}
        productID = helper.registerData(crawlSearchID, site, standardUrl, referenceOrder, trendOrder, uniq_params, params)
        productIDs.append(productID)
    return productIDs

