# https://github.com/cvhau/pinterest-client/blob/master/pinterest/Pinterest.py
# https://github.com/bstoilov/py3-pinterest
import json
import os
import time
import datetime
import requests
import requests.cookies
import numpy as np
from bs4 import BeautifulSoup, ResultSet
from requests.structures import CaseInsensitiveDict
from WebCrawlers.PinterestCrawler.registry import Registry
from WebCrawlers.PinterestCrawler.exceptions import PinterestLoginFailedException, PinterestLoginRequiredException, PinterestException
from WebCrawlers.PinterestCrawler.RequestBuilder import RequestBuilder
from WebCrawlers.PinterestCrawler.BookmarkManager import BookmarkManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import helper_functions
import pandas as pd
import sqlalchemy

import config
import urllib	

AGENT_STRING = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36"

class PinterestScraper:
    """
    Home page url needs to be modified as per user's location
    If not, the scraping following the user's login falls in a loop of redirects
    """
    home_page = 'https://gr.pinterest.com/'

    def __init__(self, username_or_email, password, proxies=None, agent_string=None):
        self.debug = False
        self.is_logged_in = False
        self.user = None
        self.req_builder = RequestBuilder()
        self.bookmark_manager = BookmarkManager()
        self.http = requests.session()
        self.username_or_email = username_or_email
        self.password = password
        self.proxies = proxies
        self.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'scraper_data', self.username_or_email) + os.sep
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        if 'registry.dat' in os.listdir(self.data_path):
            print('Refreshing \"registry.dat\"')
            os.remove(os.path.join(self.data_path, 'registry.dat'))
        self.registry = Registry('%sregistry.dat' % self.data_path)
        if agent_string:
            self.registry.set(Registry.Key.USER_AGENT, agent_string)
        elif not self.registry.get(Registry.Key.USER_AGENT):
            self.registry.set(Registry.Key.USER_AGENT, AGENT_STRING)
        old_cookies = self.registry.get(Registry.Key.COOKIES)
        if old_cookies:
            self.http.cookies.update(old_cookies)
        self.next_book_marks = {'pins': {}}
        # Load custom attributes from fashio word list and custom attributes used in the image annotation module:
        # Fashion attributes
        file_path = os.path.join(helper_functions.WEB_CRAWLERS, 'PinterestCrawler', 'fashion_words_700.txt')
        fashion_att_file = open(file_path, "r")
        fashion_att = fashion_att_file.read().split(',')
        fashion_att_file.close()
        # Image annotation labels
        fashionLabels = pd.read_excel(config.PRODUCT_ATTRIBUTES_PATH, sheet_name=config.SHEETNAME)
        attributList = np.hstack([fashionLabels[attr].replace(' ', np.nan).dropna().unique() for attr in config.PRODUCT_ATTRIBUTES]).tolist()
        fashion_att = helper_functions.preprocess_words(fashion_att + attributList)
        self.fashion_att = fashion_att

    def url_encode(self, query):	
        if isinstance(query, str):	
         query = urllib.parse.quote_plus(query)	
        else:	
            query = urllib.parse.urlencode(query)	
        query = query.replace('+', '%20')	
        return query

    def request(self, method, url, params=None, data=None, files=None, headers=None, ajax=False, stream=None):
        """
        :rtype: requests.models.Response
        """
        _headers = CaseInsensitiveDict([
            ('Accept', 'text/html,image/webp,image/apng,*/*;q=0.8'),
            ('Accept-Encoding', 'gzip, deflate'),
            ('Accept-Language', 'en-GB,en-US;q=0.9,en;q=0.8'),
            ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7'),
            ('Cache-Control', 'no-cache'),
            ('Connection', 'keep-alive'),
            ('Host', 'gr.pinterest.com'),
            ('Origin', 'https://gr.pinterest.com'),
            ('Referer', self.home_page),
            ('User-Agent', self.registry.get(Registry.Key.USER_AGENT))])
        if method.upper() == 'POST':
            _headers.update([('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')])
        if ajax:
            _headers.update([('Accept', 'application/json')])
            csrftoken = self.http.cookies.get('csrftoken')
            if csrftoken:
                _headers.update([('X-CSRFToken', csrftoken)])
            _headers.update([('X-Requested-With', 'XMLHttpRequest')])
        if headers:
            _headers.update(headers)
        response = self.http.request(method, url, params=params, data=data, headers=_headers,
                                     files=files, timeout=60, proxies=self.proxies, stream=stream)
        response.raise_for_status()
        # Handle redirection
        if response.history:
            print('Request redirected to %s ' % (response.url))
            response = requests.get(response.url, headers=_headers)

        self.registry.update(Registry.Key.COOKIES, response.cookies)
        return response

    def get(self, url, params=None, headers=None, ajax=False, stream=None):
        """
        :rtype: requests.models.Response
        """
        return self.request('GET', url=url, params=params, headers=headers, ajax=ajax, stream=stream)

    def post(self, url, data=None, files=None, headers=None, ajax=False, stream=None):
        """
        :rtype: requests.models.Response
        """
        return self.request('POST', url=url, data=data, files=files, headers=headers, ajax=ajax, stream=stream)
    
    def extract_user_data(self, html_page=''):
        """
        Extract user data from html page if available otherwise return None
        :rtype: dict|None
        """
        if html_page:
            s = html_page[html_page.rfind('application/json'):]
            if s and (s.rfind(self.username_or_email) > -1):
                s = s[s.find('{'): s.find('</script>')]
                s = json.loads(s)
                try:
                    user = s['context']['user']
                    return user
                except KeyError:
                    pass
        return None
    
    def login(self):
        """
        Login to pinterest site. If OK return True
        :rtype: bool
        """
        r = self.get(self.home_page)
        self.user = self.extract_user_data(r.text)
        time.sleep(2)
        self.login_page = 'https://gr.pinterest.com/login/?referrer=home_page'
        self.get(self.login_page)
        time.sleep(3)
        data = self.url_encode({
            'source_url': '/login/?referrer=home_page',
            'data': json.dumps({
                'options': {'username_or_email': self.username_or_email, 'password': self.password},
                "context": {}
            }).replace(' ', '')
        })
        url = 'https://gr.pinterest.com/resource/UserSessionResource/create/'
        result = self.post(url=url, data=data, ajax=True).json()
        if result['resource_response']['code'] == 0:
            self.user = self.extract_user_data(str(self.get(self.home_page).content))
            self.is_logged_in = True
        else:
            raise PinterestLoginFailedException('[%s Login failed] %s' % (result['resource_response']['http_status'], result['resource_response']['message']))
        return self.is_logged_in
        
    def login_required(self):
        if not self.is_logged_in:
            raise PinterestLoginRequiredException("Login is required")
    
    def search(self, scope, query, next_page=True):
  
        next_bookmark = self.bookmark_manager.get_bookmark(primary='search', secondary=query)

        if next_bookmark == '-end-':
            return []

        page_size=250
        terms = query.split(' ')
        escaped_query = "%20".join(terms)
        term_meta_arr = []
        for t in terms:
            term_meta_arr.append('term_meta[]=' + t)
        term_arg = "%7Ctyped&".join(term_meta_arr)
        source_url = '/search/{}/?q={}&rs=typed&{}%7Ctyped'.format(scope, escaped_query, term_arg)
        options = {
            "isPrefetch": False,
            "auto_correction_disabled": False,
            "query": query,
            "redux_normalize_feed": True,
            "rs": "typed",
            "scope": scope,
            "page_size": page_size,
            "bookmarks": [next_bookmark]
        }        

        BASE_SEARCH_RESOURCE = 'https://gr.pinterest.com/resource/BaseSearchResource/get'

        url = self.req_builder.buildGet(url=BASE_SEARCH_RESOURCE, options=options, source_url=source_url)
        r = self.get(url=url)

        jsonInfo = r.json()
        results, bookmark = [], []
        try:
            bookmark = jsonInfo['resource']['options']['bookmarks'][0]
            results = jsonInfo['resource_response']['data']['results']
        except KeyError:
            pass
        
        self.bookmark_manager.add_bookmark(primary='search', secondary=query, bookmark=bookmark)
        return results 

    def __search_next_page(self, scope, query):
        q = self.url_encode({
            'source_url': '/search/%s/?q=%s' % (scope, query),
            'data': json.dumps({
                'options': {
                    'bookmarks': [self.next_book_marks[scope][query]],
                    'query': query,
                    'scope': scope
                },
                "context": {}
            }).replace(' ', ''),
            '_': '%s' % int(time.time() * 1000)
        })

        url = 'https://gr.pinterest.com/resource/SearchResource/get/?%s' % q
        r = self.get(url=url, ajax=True).json()
        results = []
        try:
            if r['resource_response']['status'] != 'success':
                raise PinterestException('[%s] %s' % (r['resource_response']['http_status'], r['resource_response']['message']))
            results = r['resource_response']['data']
            bookmarks = r['resource']['options']['bookmarks']
            if isinstance(bookmarks, str):
                self.next_book_marks[scope][query] = bookmarks
            else:
                self.next_book_marks[scope][query] = bookmarks[0]
        except KeyError:
            pass
        return results

    def search_pins(self, query, next_page=True, threshold=10):
        self.login_required()
        engine = helper_functions.ENGINE

        # SELECTSQLQUERY = '''SELECT *
        # 	                FROM S4F.dbo.Product'''
        SELECTSQLQUERY = '''SELECT *
        	                FROM public.\"Product\"'''

        productsDF = pd.read_sql_query(SELECTSQLQUERY, engine)
        pins = []
        results = self.search('pins', query, next_page=next_page)
        for index, result in enumerate(results):
            if result['type'] == 'pin' and result['videos'] == None and threshold > 0:
                try:
                    tempDF = productsDF.loc[productsDF['URL'] == result['link']]
                    promotion = result['is_promoted']
                    if tempDF.empty and not promotion:
                        searchwords = ''.join(query.split(" "))
                        imageFilePath = helper_functions.setImageFilePath(self.login_page, searchwords, index)
                        # Capture the main attributes of the result to evaluate and enchance the final recommendation
                        pins.append({'query':query,
                                     'timestamp': str(datetime.datetime.now()),
                                     'URL': result['link'],
                                     'imgURL':result['images']['orig']['url'], 
                                     'imageFilePath':imageFilePath, 
                                     'title':result['title'], 
                                     'description':result['description']})

                        # Create new entry in ProductHistory table
                        # helper_functions.addNewProductHistory(result['link'], index, 0, 0)
                        threshold -= 1
                        if threshold<0:
                            break
                except TypeError:
                    pass
        return pins

    def rankedSave(self, pins):
        pinsDF = pd.DataFrame(pins)
        # Form metadata column
        pinsDF['metadata'] = pinsDF['title'].str.cat(pinsDF['description'].astype(str), sep=' ')
        # Preprocess metadata
        pinsDF['processed_metadata'] = pinsDF['metadata'].apply(helper_functions.preprocess_metadata)
        # Preprocess query
        pinsDF['query'] = pinsDF['query'].apply(lambda row: ' '.join(helper_functions.preprocess_words(row.split())))

        ## Calculate a factor for tokens that appear in metatdata
        keywords = pins[0]['query']
        pinsDF['factor'] = pinsDF['processed_metadata'].apply(lambda row: len(set([word for word in row if word in keywords]))/len(keywords))

        ## Calculate a factor based on the cosine similarity of TFIDF transformation of the query terms and 
        # the processed metadata using the fashion expert terminology as vocabulary
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,1))
        vectorizer.fit_transform(self.fashion_att)
        tfidf_vector = vectorizer.transform(pinsDF['processed_metadata'])
        query_vector = vectorizer.transform(pinsDF['query'])

        ## Calculate cosine similarity
        cosine_vector = cosine_similarity(query_vector[0], tfidf_vector)
        pinsDF['cosine ranking score'] = np.hstack(cosine_vector).tolist() * pinsDF['factor']

        ## Calculate a factor based on Pinterest's recommendation (order of result parsing)
        scaler = MinMaxScaler()
        pinterest_score = scaler.fit_transform(np.arange(len(pinsDF)).reshape(-1, 1))
        pinsDF.loc[pinsDF.sort_values(by ='timestamp', ascending=False).index, 'pinterest score'] = pinterest_score
        
        ## Calculate Final Ranking Score giving the cosine similarity factor a greater score than the 
        # factor based on the Pinterest recommendation
        pinsDF['final score'] = (pinsDF['cosine ranking score'] * 0.7) + (pinsDF['pinterest score'] * 0.3)
        pinsDF.sort_values(by ='final score', ascending=False, inplace=True)

        # Save ranked results to the database
        for _, row in pinsDF.iterrows():
            site = self.login_page
            searchwords = ''.join(keywords.split())
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



