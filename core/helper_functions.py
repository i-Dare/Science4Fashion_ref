# -*- coding: utf-8 -*-
############################# Import all the libraries needed ###########################
import os
import requests
import time
import random
import json
import string
import cv2
import pandas as pd
import regex as re
import numpy as np
from PIL import Image as PILImage
from fastai.vision import *
from bs4 import BeautifulSoup, ResultSet
from datetime import datetime
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import wordsegment

from pathlib import Path
import warnings

import core.config as config
from core.query_manager import QueryManager

wordsegment.load()
warnings.filterwarnings('ignore')


class Helper():
    # Define stop words
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
    STOP_WORDS = STOP_WORDS.union(set(nltk.corpus.stopwords.words('italian')),
                                set(nltk.corpus.stopwords.words('german')),
                                set(nltk.corpus.stopwords.words('french')),
                                set(nltk.corpus.stopwords.words('spanish')))
    # Removed 'man', included in german stop_words as it is an English word
    STOP_WORDS.remove('man')
    # Add 'via' in stop_words
    STOP_WORDS.add('via')

    def __init__(self, logging):
        self.logging = logging
        self.logger = logging.logger
        self.user = logging.user
        self.db_manager = QueryManager(user=self.user)
        
# --------------------------------------------------------------------------                        
#          Miscellaneous Functionality
# -------------------------------------------------------------------------- 

    ## This function will create a soup and returns which is the parsed html format for extracting 
    # html tags of the webpage 
    def get_content(self, url, suffix='', retry=3):
        for i in range(1, retry):
            user_agent_list = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
            ]
            # Setup header request with random User-Agent
            user_agent = random.choice(user_agent_list)
            headers = {
                'Referer': url,
                'User-Agent': user_agent
            }
            self.logger.info('Get content from: %s, try: %s ' % (url+suffix, i))
            try:
                req = requests.get(url+suffix, headers=headers, timeout=config.CRAWLER_TIMEOUT ,verify=False)
                soup = BeautifulSoup(req.content, 'html.parser')
                if req.reason == 'OK':
                    url = req.url
                    return url, soup
                
                new_req = ''            
                if req.history:
                    self.logger.info('Request redirected to %s try: %s ' % (url+suffix, i))
                    new_req = requests.get(req.url+suffix, headers=headers, timeout=config.CRAWLER_TIMEOUT, 
                            verify=False)
                    soup = BeautifulSoup(new_req.content, 'html.parser')                    
                    if new_req.reason == 'OK':
                        url = new_req.url
                        return url, soup
            except requests.exceptions.Timeout:
                self.logger.warning("Timeout occurred when getting %s try: %s " % (url+suffix, i))
        return None, None
        

    ## This function returns the folder name removing the number of images range from they line of 
    # keywords file 
    #
    def getFolderName(self, wholeName):
        wholeName = re.sub(r'[0-9]', '', wholeName).strip()
        argList = wholeName.split(' ')
        if len(argList)==1:
            return  argList[0]
        return ''.join(argList[1:]) if type(argList[1])==int else ''.join(argList)
        

    ##  This function returns the 3rd appearance of "/""
    #
    def hyphen_split(self, a):
        if a.count("/") == 1:
            return a.split("/")[0]
        else:
            return "/".join(a.split("/", 3)[:3])


    ## Call this function for each image for each model to extract the predicted label
    #
    def updateAttribute(self, modeldictionary, image, model):
        x = pil2tensor(image, np.float32)
        x = x/255
        _, pred_idx, _ = model.predict(Image(x))
        #Convert label from model to database format
        label = modeldictionary[int(pred_idx.numpy())]
        return label


    ## Convert digital data to binary format 
    #
    def convertToBinaryData(self, filename):
        '''
        Convert digital data to binary format
        '''
        with open(filename, 'rb') as file:
            blobData = file.read()
        return blobData


    ## Convert imageBlob to img
    #
    def convertBlobToImage(self, imageBlob):
        '''
        Converts byte stream to 3D numpy array
        '''
        imgArray = np.frombuffer(imageBlob, dtype='uint8')
        # decode the array into an image
        return cv2.imdecode(imgArray, cv2.IMREAD_UNCHANGED)


    def convertCVtoPIL(self, img):
        ###Convert image fron OpenCV to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(img)
        return im_pil    


    def openUnicodeImgPath(self, imgPath):
        imgArray = np.fromfile(imgPath, dtype=np.uint8)
        return cv2.imdecode(imgArray, cv2.IMREAD_UNCHANGED)

    ## Set image destination path 
    #
    def setImageFilePath(self, standardUrl, keyword, i):
        '''
        Sets image destination path.
        - Input:
            standardUrl: base website URL
            keyword: search term
            i: iteration counter

        - Returns:
            imageFilePath: image destination path
        '''
        dt_string = datetime.now().strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
        nameOfFile = 'image' + str(i) + "_" + re.sub("[ /:]", ".", dt_string)
        ImageType = ".jpeg"
        imgSiteFolder = (standardUrl.split('.')[1]).capitalize() + 'Images'
        # Creates the path where the images will be stored
        imageDir = os.path.join(config.IMAGEDIR, imgSiteFolder, keyword)
        # Create The folder according to search name
        if not os.path.exists(imageDir):
            os.makedirs(imageDir)
        imageFilePath = os.path.join(imageDir, nameOfFile + ImageType)
        return imageFilePath


    def getWebImage(self, imgURL):
        from urllib.request import urlopen
        resp = urlopen(imgURL)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        return cv2.imdecode(image, cv2.IMREAD_COLOR)


    def saveImage(self, imgURL, imageFilePath):
        '''
        Retrieves image from a URL and saves it to given file path and returns its binary data 
        representation
        - Input:
            imgURL: image URI
            imageFilePath: image destination directory

        - Returns:
            empPhoto: image binary
        '''
        r = requests.get(imgURL, allow_redirects=True)

        imageFile = open(imageFilePath, 'wb')
        imageFile.write(r.content)
        imageFile.close()
        # convert image to binary
        empPhoto = self.convertToBinaryData(imageFilePath)
        return empPhoto

    def save_model(self, model, model_name, directory):
        '''
        Saves model locally in a pickle
        '''
        # Create model directory
        if not os.path.exists(directory):      
            os.makedirs(directory)
        extension = '' if model_name.endswith('.pkl') else '.pkl'
        model_name += extension
        model_path = os.path.join(directory, model_name)
        pickle.dump(model, open(model_path, 'wb'))
        self.logger.info('Model %s saved at %s' % (model_name, model_path))

    def get_model(self, model_name):
        try:
            model_list_paths = [str(p.joinpath()) for p in Path(config.MODELSDIR).rglob('*.pkl') 
                    if model_name in str(p.joinpath())]

            model_path = max(model_list_paths, key = os.path.getctime)
            model = pickle.load(open(str(model_path).replace('\\', os.sep), 'rb'))
            return model
        except:
            self.logger.warning('Model %s not found in directory %s.' % (model_name, config.MODELSDIR))

    def remove_model(self, model_name):
        try:
            model_list_paths = [str(p.joinpath()) for p in Path(config.MODELSDIR).rglob('*.pkl') 
                    if model_name in str(p.joinpath())]

            model_path = max(model_list_paths, key = os.path.getctime)
            os.remove(str(model_path).replace('\\', os.sep))
        except:
            self.logger.warning('Model %s not found in directory %s.' % (model_name, config.MODELSDIR))
    

# --------------------------------------------------------------------------                        
#          Database IO Functionality
# -------------------------------------------------------------------------- 

    ## Return gender ID according to the extracted text from the crawler
    #
    def getGender(self, gender):
        gender_dict = {
            'man': ['man', 'men', 'male'],
            'woman': ['woman', 'women', 'female'],
            'kids': ['kid', 'kids', 'child', 'children', 'junior'],
            'unisex': ['unisex']}

        genderdf = pd.read_sql_query("SELECT * FROM %s.dbo.Gender" % config.DB_NAME, config.ENGINE)
        genderdf_dict = genderdf.set_index('Oid').loc[:, 'Description'].str.lower().to_dict()

        # Find gender in gender_dict
        gnd = [g_label for g_label, g in gender_dict.items() if gender.lower() in g][0]
        # Find captured gender in gender dataframe
        gender_id = [g_id for g_id, g in genderdf_dict.items() if gnd==g][0]
        return gender_id   


    ## Add new Brand to the Brand table 
    #
    def getBrand(self, brand):
        '''
        Adds new brand info to the Brand table if it does not exist,  and returns the added record's 
        `Oid`. Otherwise returns the existing record 'Oid'. 
        '''
        uniq_params = {'table': 'Brand', 'Description': brand}
        params = {'table': 'Brand', 'Description': brand}
        
        brand_df =  self.db_manager.runCriteriaInsertQuery(
                                                            uniq_params=uniq_params, 
                                                            params=params, 
                                                            get_identity=True
                                                        )
        return brand_df.loc[0, 'Oid']

    ## Add new ProductCategory to the ProductCategory table 
    #
    def getProductCat(self, prodCat):
        '''
        Adds new ProductCategory info to the ProductCategory table if it does not exist,  
        and returns the added record's  `Oid`. Otherwise returns the existing record 'Oid'. 
        '''
        uniq_params = {'table': 'ProductCategory', 'Description': prodCat}
        params = {'table': 'ProductCategory', 'Description': prodCat}
        
        prodCat_df =  self.db_manager.runCriteriaInsertQuery(
                                                            uniq_params=uniq_params, 
                                                            params=params, 
                                                            get_identity=True
                                                        )
        return prodCat_df.loc[0, 'Oid'] 
    
    def getProductSubCat(self,  prodSubCat, prodCat):
        '''
        Adds new ProductSubcategory info to the ProductSubcategory table if it does not exist,  
        and returns the added record's `Oid`. Otherwise returns the existing record 'Oid'. 
        Receives the product category and subCategory as arguments. If the product category is a 
        string, it is used to find the respective category from ProductCategory table, if it is an
        integer, it is considered as productCategory Oid.

        '''        
        if type(prodCat) == str:
            prodCatID = self.getProductCat(prodCat)
        elif type(prodCat) == int:
            prodCatID = prodCat
        else:
            return None

        uniq_params = {'table': 'ProductSubcategory', 
                        'Description': prodSubCat, 
                        'ProductCategory': prodCatID}

        params = {'table': 'ProductSubcategory', 
                  'Description': prodSubCat, 
                  'ProductCategory': prodCatID}
        
        prodSubCat_df =  self.db_manager.runCriteriaInsertQuery(
                                                            uniq_params=uniq_params, 
                                                            params=params, 
                                                            get_identity=True
                                                        )
        return prodSubCat_df.loc[0, 'Oid']
        

    ## Add new product to the Product table 
    #
    def addNewProduct(self, site, uniq_params, params):
        '''
        Adds new product info to the Product table. 
        '''
        params['Adapter'] = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter WHERE \
                %s.dbo.Adapter.Description = '%s'" % (config.DB_NAME, config.DB_NAME, site), 
                config.ENGINE).loc[0, 'Oid']
        # Add a new Brand if it does not exist dn return the 'Oid' or just return existing 'Oid'
        if 'Brand' in params.keys():
            params['Brand'] = self.getBrand(params['Brand'])        
        
        ## Add product and return new records in DataFrame
        # Check if product exists
        # If DataFrame is empty add new product
        self.logger.info('Adding new product')
        product_df = self.db_manager.runCriteriaInsertQuery(
                                                            uniq_params=uniq_params, 
                                                            params=params, 
                                                            get_identity=True
                                                        )
        return product_df
  

    ## Add new product to the ProductHistory table 
    #
    def addNewProductHistory(self, product_df, referenceOrder, trendOrder):
        '''
        Adds new product info to the ProductHistory table.
        '''
        oid, price, searchTerm, adapter = product_df.loc[0, ['Oid', 'RetailPrice', 'Description', 'Adapter']]
        ## If entry in ProductHistory for Product 'Oid' and CrawlSearch 'Oid' does not exist, create new entry
        # Get latest 'Oid' from CrawlSearch table for the specific search term and user
        lastCrawlSearchID = self.db_manager.getLastRecordID('CrawlSearch', "WHERE Description='%s' \
                AND CreatedBy='%s' AND Adapter=%s" % (searchTerm, self.user, adapter))
        
        self.logger.info('Adding new product history')
        uniq_params = {'table': 'ProductHistory', 'Product': oid, 'CrawlSearch': lastCrawlSearchID}
        params = {'table': 'ProductHistory', 'Product': oid, 'ReferenceOrder': referenceOrder, 
                  'TrendingOrder': trendOrder, 'Price': price, 'CrawlSearch': lastCrawlSearchID}
        return self.db_manager.runCriteriaInsertQuery(uniq_params=uniq_params, params=params, get_identity=True)

    ## Update product history 
    #
    def updateProductHistory(self, product_df, referenceOrder, trendOrder):
        '''
        Updates product's latest entry in ProductHistory table
        '''    
        productID, price, searchTerm, adapter = product_df.loc[0, ['Oid', 'RetailPrice', 'Description', 'Adapter']]
        # Get latest 'Oid' from CrawlSearch table for the specific search term and adapter
        params = {'table': 'CrawlSearch', 'SearchTerm': searchTerm, 'Adapter': adapter}         
        crawlSearch_df = self.db_manager.runSelectQuery(params)
        crawlSearch_df.sort_values('Oid', ascending=False, inplace=True)
        # Get all productHistory for the specific product
        params = {'table': 'ProductHistory', 'Product': productID}
        productHist_df = self.db_manager.runSelectQuery(params)
        # Select the relevant CrawlSerach 'Oid' as the latest (max) 'Oid' in the retrieved ProductHistory
        lastCrawlSearchID = crawlSearch_df[crawlSearch_df['Oid'].isin(productHist_df['CrawlSearch'])]['Oid'].max()

        uniq_params = {'table': 'ProductHistory', 'Product': productID, 'CrawlSearch': lastCrawlSearchID}
        params = {'table': 'ProductHistory', 'UpdatedBy': self.user, 'Price': price, 'TrendingOrder': trendOrder,  
                  'ReferenceOrder': referenceOrder, 'CrawlSearch': lastCrawlSearchID}
        return self.db_manager.runCriteriaUpdateQuery(uniq_params=uniq_params, params=params, get_identity=True)
        

# --------------------------------------------------------------------------
#          Natural Language Processing (NLP) Functionality 
# --------------------------------------------------------------------------    

    def get_fashion_attributes(self, ):
        ## Load custom attributes from fashion word list and custom attributes used in the image 
        # annotation module
        file_path = config.FASHION_WORDS
        fashion_att_file = open(file_path, "r")
        fashion_att = fashion_att_file.read().split(',')
        fashion_att_file.close()
        # Image annotation labels
        fashionLabels = pd.read_excel(config.PRODUCT_ATTRIBUTES_PATH, sheet_name=config.SHEETNAME)
        attributList = np.hstack([fashionLabels[attr].replace(' ', np.nan).dropna().unique() for attr in config.PRODUCT_ATTRIBUTES]).tolist()
        fashion_att = self.preprocess_words(fashion_att + attributList)
        return fashion_att


    def preprocess_words(self, words_list):
        lemmatizer = WordNetLemmatizer()
        preprocessed_words = []
        for word in words_list:
            preprocessed_words.append(lemmatizer.lemmatize(word.lower(), "n"))
        return preprocessed_words


    def lemmatize(self, token, pos_tag):
        lemmatizer = WordNetLemmatizer()
        tag = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}.get(pos_tag[0], wn.NOUN)
        return lemmatizer.lemmatize(token, tag)


    def preprocess_metadata(self, doc, segmentation=False):

        # Convert to lowercase
        doc = doc.lower()
        # Remove URLs
        doc = re.sub(r'(www\S+)*(.\S+\.com)', '', doc)
        # Word segmentation, used for compound words, hashtags and spelling errors
        if segmentation:
            doc = ' '.join(wordsegment.segment(doc))
        # Remove punctuation
        doc = re.sub('[' + re.escape(string.punctuation.replace('_', '')) + ']+', ' ', doc)
        # Remove two letter words
        doc = ' '.join([word for word in doc.split() if len(word)>2])
        # Remove numbers and words with number
        doc = re.sub(r'([a-z]*[0-9]+[a-z]*)', '', doc)
        # Remove non-ASCII characters 
        doc = str(doc).encode("ascii", errors="ignore").decode()
        # Remove excess whitespace
        doc = re.sub(r'\s+', ' ', doc)
        # Remove stop words
        doc = ' '.join([word for word in doc.split() if word not in self.STOP_WORDS])
        
        # Tokenize
        tokenizer = TweetTokenizer(reduce_len=True)
        tokens = tokenizer.tokenize(doc)
        # Lemmatize
        tokens = [self.lemmatize(word, tag) for word,tag in pos_tag(tokens)]
        # Merge together
        return ' '.join(tokens)

    def create_tfidf_features(self, vocabulary, doc):

        # Setup TFIDF vectorizer
        vectorizer = TfidfVectorizer(analyzer='word', 
                                     ngram_range=(1,2), 
                                     min_df=2, 
                                     stop_words=self.STOP_WORDS)

        vectorizer.fit_transform(vocabulary)
        tfidf_vector = vectorizer.transform(doc)
        return tfidf_vector

# --------------------------------------------------------------------------                        
#          Web crawler functionality, specific for each website
# --------------------------------------------------------------------------
    ## Generic functionaloty
    # Add/update product information
    def registerData(self, site, standardUrl, referenceOrder, trendOrder, cnt, uniq_params, params):
        from ImageAnnotation.color import ColorAnnotation
        from ImageAnnotation.clothing import ClothingAnnotation
        from TextMining import MetadataAnnotation

        # Check if product url exists to decide if addition of update is needed
        url = params['URL']
        _params = {'table': 'Product', 'URL': url}
        product_df = self.db_manager.runSelectQuery(_params)

        if not product_df.empty:
            action = 'update' # action flag
            ## Product exists, proceed to update ProductHistory table
            # Update ProductHistory
            try:
                productHist_df = self.updateProductHistory(product_df, referenceOrder, trendOrder)
                cnt += 1
            except Exception as e:
                self.logger.warn_and_trace(e)
            
        else:
            action = 'addition' # action flag
            # Download product image
            params['Photo'] = self.setImageFilePath(standardUrl, 
                    ''.join(params['Description'].split()), trendOrder)
            params['Image'] = self.saveImage(params['ImageSource'], params['Photo'])
            try:
                # Create new entry in PRODUCT table
                product_df = self.addNewProduct(site, uniq_params=uniq_params, params=params)

                cnt += 1
                # Create new entry in ProductHistory table
                productHist_df = self.addNewProductHistory(product_df, referenceOrder, trendOrder)
                # Product IDs to perform annotation
                oids = product_df.loc[:, 'Oid'].tolist()
                # Color annotation                
                color_annotator = ColorAnnotation.ColorAnnotator(self.user, *oids)
                color_annotator.execute_annotation()
                # Clothing annotation                
                clothing_annotator = ClothingAnnotation.ClothingAnnotator(self.user, *oids)
                clothing_annotator.execute_annotation()
                # Metatada annotation                
                metadata_annotator = MetadataAnnotation.MetadataAnnotator(self.user, *oids)
                metadata_annotator.execute_annotation()

            except Exception as e:
                self.logger.warn_and_trace(e)       
                self.logger.warning('Information for product at %s not added' % \
                        re.findall(r'\'(http.+)\',', params['URL'])[0])
                return cnt, None
            
        productID = productHist_df.loc[0, 'Product']
        if action == 'addition':           
            self.logger.info('Information for product %s added' % productID)
        else: 
            self.logger.info('Information for product %s updated' % productID)
        return cnt, productID

    ## Asos specific functionality 
    ## Fetch search results form Asos according to 'order' parameter 
    #
    def resultDataframeAsos(self, keyUrl, order, filterDF=pd.DataFrame([]), breakPointNumber=9999999):
        '''
        Fetches search results form Asos according to 'order' parameter. Parameter "filterDF" is 
        filled with the trending order dataframe to call the script with reference order and get the reference order
        only for the trending items.
        - Input:
            keyUrl: search URL according to keywords
            order: ordering of results, valid values are 'reference' and 'trend'

        - Returns:
            resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
        '''
        # initialize scraping
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None
        pageNo = 1
        # results' ordering
        rule = '' if ordering == orderDict['reference'] else '&sort=freshness'
        paging = '&page=%s' % pageNo
        resultsURL = keyUrl + rule + paging

        # Fetch initial search result page
        url, soup = self.get_content(resultsURL, retry=5)
        maxItems = int(soup.find('progress').get('max'))
        breakPointNumber = breakPointNumber if breakPointNumber <= maxItems else maxItems
        # Prepare ajax request URL
        # Capture hidden parameters of ajax request for the image loading
        try:
            jsonUnparsed = [script for script in soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in str(script)]
            jsonInfo = json.loads(re.findall(r'JSON\.parse.+({\"(?=analytics).+products\".+\]}}})', str(jsonUnparsed))[0].replace('\\', ''))
            articles = jsonInfo['search']['products']
            products = articles
        except Exception as e:        
            self.logger.warn_and_trace(e)

        self.logger.info('Prepare to fetch results according to %s order' % order)
        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL', 'price'])
        if order=='reference':
            breakPointNumber = len(filterDF)
            # Search for URLs if they exist in the filterDF
            while len(products) < maxItems:
                urlList = ['https://www.asos.com/uk/' + a['url'] for a in articles]
                if not filterDF[filterDF['URL'].isin(urlList)].empty:
                    resultsDF = resultsDF.append(filterDF[filterDF['URL'].isin(urlList)], 
                            ignore_index=True)
                if len(resultsDF) >= len(filterDF): # Exit condition for reference order
                    resultsDF = resultsDF[:len(filterDF)]
                    break
                pageNo += 1
                _paging = '&page=%s' % pageNo
                _resultsURL = keyUrl + rule + _paging
                url, _soup = self.get_content(_resultsURL, retry=5)
                try:
                    # Prepare ajax request URL
                    # Capture hidden parameters of ajax request for the image loading
                    _jsonUnparsed = [script for script in _soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in str(script)]
                    _jsonInfo = json.loads(re.findall(r'JSON\.parse.+({\"(?=analytics).+products\".+\]}}})', 
                            str(_jsonUnparsed))[0].replace('\\', ''))
                    articles = _jsonInfo['search']['products']
                    products += articles
                    if len(articles)==0:
                        break
                except Exception as e:                
                    self.logger.warn_and_trace(e)
                    self.logger.warning('Failed to parse %s' % resultsURL)
        else:
            maxItems = breakPointNumber
            while len(products) < maxItems:
                pageNo += 1
                _paging = '&page=%s' % pageNo
                _resultsURL = keyUrl + rule + _paging
                url, _soup = self.get_content(_resultsURL, retry=5)
                try:
                    # Prepare ajax request URL
                    # Capture hidden parameters of ajax request for the image loading
                    _jsonUnparsed = [script for script in _soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in str(script)]
                    _jsonInfo = json.loads(re.findall(r'JSON\.parse.+({\"(?=analytics).+products\".+\]}}})', 
                            str(_jsonUnparsed))[0].replace('\\', ''))
                    articles = _jsonInfo['search']['products']
                    if len(articles)==0:
                        break
                except Exception as e:                
                    self.logger.warn_and_trace(e)
                    self.logger.warning('Failed to parse %s' % resultsURL)
                products += articles
            products = products[:maxItems]  
            # Iterate result pages
            for product in products:
                # Capture actual products information from json fields
                series = pd.Series([])

                productPage = product['url']
                productPage = 'https://www.asos.com/uk/' + productPage
                productImg = 'https://' + product['image']

                # Get product price
                try:
                    price = product['price']
                    sales_price = product['reducedPrice']
                    isSale = product['isSale']
                    price = sales_price if isSale else price
                except:
                    price = None
                    self.logger.info("Price not captured for %s" % productPage)

                series = pd.Series({'URL': productPage,
                                    'imgURL': productImg,
                                    'price': price},
                                index=resultsDF.columns)
                resultsDF = resultsDF.append(series, ignore_index=True)
        resultsDF[ordering] = range(1, len(resultsDF)+1)
        return resultsDF


    ## Get product information from Asos 
    #
    def parseAsosFields(self, soup, url):
        '''
        Gets product information from Asos
        - Input:
            soup: BeautifulSoup navigator
            url: URL currently parsing

        - Returns:
            head, brand, color, genderid, meta
        '''
        # Capture json structure with the product information
        try:
            jsonUnparsed = soup.find('script', text=re.compile(r'window\.asos\.pdp\.config\.product.+'))
            jsonInfo = json.loads(re.findall(r'window\.asos\.pdp\.config\.product = ({.+});', str(jsonUnparsed))[0])
        except Exception as e: 
            self.logger.warn_and_trace(e)
            self.logger.warning('Exception: Failed to capture json object with products\' info')
        # Get attributes
        # Header
        try:
            head = jsonInfo['name']
        except Exception as e:
            head = None 
            self.logger.warn_and_trace(e)
            self.logger.warning("Header not captured at %s" % url)
        # Sku ID
        try:
            sku = jsonInfo['productCode']
        except Exception as e: 
            sku = None
            self.logger.warn_and_trace(e)
            self.logger.warning("SKU ID not captured at %s" % url) 
        # Image
        try:
            imgURL = jsonInfo['images'][0]['url']
        except Exception as e: 
            imgURL = None
            self.logger.warn_and_trace(e)
            self.logger.warning("Image url not captured at %s" % url) 
        # Gender
        try:
            gender = jsonInfo['gender']
            genderid = self.getGender(gender)

        except Exception as e: 
            genderid = None
            self.logger.warn_and_trace(e)
            self.logger.warning("GenderID not captured at %s" % url)    
        # Brand
        try:
            brand = jsonInfo['brandName']
        except Exception as e: 
            brand = None
            self.logger.warn_and_trace(e)
            self.logger.warning("Brand info not captured at %s" % url)    
        # Color
        try:
            color = jsonInfo['variants'][0]['colour']
        except Exception as e: 
            color = None
            self.logger.warn_and_trace(e)
            self.logger.warning("Color info not captured at %s" % url)   
        # Description
        try:
            description = [li.text.strip() for li in soup.find(
                'div', {'class': re.compile('product-description')}).find('ul').findAll('li')]
            about = [soup.find('div', {'class': re.compile(
                'about-me')}).find('p').text.strip()]
            attr_list = description + about
            meta = ' - '.join(attr_list)
            meta = self.preprocess_metadata(doc=meta)

        except Exception as e: 
            meta = ''
            self.logger.warn_and_trace(e)
            self.logger.warning("Product Description not captured at %s" % url)
        # Price
        try:
            productID = jsonInfo['id']
            priceApiURL = 'https://www.asos.com/api/product/catalogue/v3/stockprice?productIds=%s&store=ROE&currency=EUR' % productID
            time.sleep(3)  # suspend execution for 5 secs
            _, priceSoup = self.get_content(priceApiURL, retry=3)
            price = json.loads(str(priceSoup))[0]['productPrice']['current']['value']
        except:
            price = None
            self.logger.warning("Product price not captured at %s" % url)   
    

        return head, brand, color, genderid, meta, sku, price, url, imgURL


    ## sOliver specific functionality 
    ## Fetch search results form s.Oliver according to 'order' parameter 
    #
    def resultDataframeSOliver(self, keyUrl, order, filterDF=pd.DataFrame([]), breakPointNumber=9999999):
        '''
        Fetches search results form s.Oliver according to 'order' parameter. Parameter "filterDF" is 
        filled with the trending order dataframe to call the script with reference order and get the reference order
        only for the trending items.
        - Input:
            keyUrl: search URL according to keywords
            order: ordering of results, valid values are 'reference' and 'trend'
            breakPointNumber: defines the number of results to fetch

        - Returns:
            resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
        '''
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None

        # Fetch initial search result page
        url, soup = self.get_content(keyUrl)

        # Initialize parameters
        start = 0  # starting point of lazy-loading
        step = 12  # lazy-lading step
        
        # Check for redirection
        if 'search' not in  url:
            self.logger.info('s.Oliver rediraction to %s' % url)
        # Prepare ajax request URL
        # Capture hidden parameters of ajax request for the image lazy-loading
        reqParams = json.loads(soup.find(
            'div', {'class': re.compile("hidden jsOVModel")}).get('data-ovmodel'))
        # Remove irrelevant parameters
        if 'qold' in reqParams['params'].keys():
            reqParams['params'].pop('qold')
        if 'autofilter' in reqParams['params'].keys():
            reqParams['params'].pop('autofilter')

        # Setup custom ajax request
        maxItems = reqParams['hitCount']  # max number of results
        resultRange = 'start=%s&sz=%s' % (start, step)  # lazy-loading URL parameter
        # results' ordering
        rule = 'srule=default&' if ordering == orderDict['reference'] else 'srule=newest-products&'
        suffix = '&lazyloadFollowing=true&view=ajax'  # ajx request suffix
        urlAjax = reqParams['urlAjax'] + '?'  # initialize ajax request
        # Appeding parameters to custom ajax request
        for k, v in reqParams['params'].items():
            urlAjax += '%s=%s&' % (k, '%20'.join(v.split(' ')))
        urlAjax += rule + resultRange + suffix

        self.logger.info('Prepare to fetch results according to %s order' % order)

        # lazy-loading iteration until reaching the maxItems - 12 products in each iteration
        url, soupAjax = self.get_content(urlAjax)
        articles = soupAjax.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
        products = articles

        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL'])
        if order=='reference':
            breakPointNumber = len(filterDF)
            while len(products) < maxItems:
                urlList = ['https://www.soliver.eu' + a.find('a', {'class': 
                            re.compile("js-ovlistview-productdetaillink")}).get('href') 
                            for a in articles]
                if not filterDF[filterDF['URL'].isin(urlList)].empty:
                    resultsDF = resultsDF.append(filterDF[filterDF['URL'].isin(urlList)], ignore_index=True)
                if len(resultsDF) >= len(filterDF): # Exit condition for reference order
                    resultsDF = resultsDF[:len(filterDF)]
                    break
                # Prepare next ajax request
                start += step
                resultRange = 'start=%s&sz=%s' % (start, step)
                urlAjax = re.sub(r'start=[0-9]+&sz=[0-9]+', resultRange, urlAjax)
                url, soupAjax = self.get_content(urlAjax)
                articles = soupAjax.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
                products += articles
                if len(articles)==0: # Exit if no more results
                    break
        else:
            maxItems = breakPointNumber
            while len(products) < maxItems:
                # Prepare next ajax request
                start += step
                resultRange = 'start=%s&sz=%s' % (start, step)
                urlAjax = re.sub(r'start=[0-9]+&sz=[0-9]+', resultRange, urlAjax)
                url, soupAjax = self.get_content(urlAjax)
                articles = soupAjax.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
                # Stop if exceed pagination
                if len(articles)==0:
                    break
                products += articles
            products = products[:maxItems]
            for product in products:
                productPage = product.find('a', {'class': re.compile(
                    "js-ovlistview-productdetaillink")}).get('href')
                productPage = 'https://www.soliver.eu' + productPage
                if product.find('li', {'class': 'plproduct__color'}):
                    productImg = (json.loads(product.find('li', {'class': 'plproduct__color'})
                            .get('data-ovlistview-color-config'))['images'][0]['absURL'])
                    color = (json.loads(product.find('li', {'class': 'plproduct__color'})
                            .get('data-ovlistview-color-config'))['colorName'])
                    price = (float(product.find('div', {'class': 'ta_prodMiniWrapper'})
                            .get('data-maxprice').split()[0].replace(',', '.')))
                else:
                    productImg = (json.loads(product.find('div', {'class': 'ta_Img'})
                            .get('data-picture'))['sources'][0]['srcset'].split('?')[0])
                    color = None
                    price = None

                series = pd.Series()
                series['URL'] = productPage.rsplit('?', 1)[0]
                series['imgURL'] = productImg
                series['color'] = color
                series['price'] = price
                resultsDF = resultsDF.append(series, ignore_index=True)
        # Set ordering
        resultsDF[ordering] = range(1, len(resultsDF)+1)
        return resultsDF


    ## Get product information from s.Oliver 
    #
    def parseSOliverFields(self, soup, url, imgURL):
        '''
        Gets product information from s.Oliver
        - Input:
            soup: BeautifulSoup navigator
            url: URL currently parsing
            imgURL: image URL

        - Returns:
            price, head, brand, color, genderid, meta
        '''
        # Get product id according to the captured image URL
        pid, sku = '', ''
        for hidden in soup.findAll('div', {'class': re.compile('hidden')}):
            if hidden.get('data-mms-spv-variationmodel'):
                jsonInfoImg = json.loads(hidden.get('data-mms-spv-variationmodel'))
                for swatch in jsonInfoImg['colorSwatches']:
                    if swatch['swatchImage']['img']['src'].split('_')[0] in imgURL:
                        pid = swatch['variantID']
                        sku = pid
                        break
                
        if sku == '':
            self.logger.info("ProductCode and active information not captured at %s" % url)
        ajaxURL = 'https://www.soliver.eu/on/demandware.store/Sites-soliverEU-Site/en/Catalog-GetProductData?pview=variant&pid='
        ajaxURL += pid
        # return ajax request for the selected image
        url, soupTemp = self.get_content(ajaxURL)
        try:
            jsonInfo = json.loads(soupTemp.text)['product']['variantInfo']
        except:
            self.logger.warn("JSON information for product at %s not found" % url)
        # color - Clothe's Color
        try:
            color = jsonInfo['variationAttrData']['color']['displayValue']
        except Exception as e:
            color = None
            self.logger.warn_and_trace(e)
            self.logger.warn("Color not captured at %s" % url)

        # price
        try:
            price = jsonInfo['priceData']['salePrice']['value']
        except Exception as e:
            price = None
            self.logger.warn_and_trace(e)
            self.logger.warn("Price not captured at %s" % url)

        # head - Clothe's General Description
        try:
            head = jsonInfo['microdata']['description']
        except Exception as e:
            head = None
            self.logger.warn_and_trace(e)
            self.logger.warn("Header not captured at %s" % url)

        # brand - Clothe's Brand
        try:
            brand = jsonInfo['microdata']['brand']
        except Exception as e:
            brand = None
            self.logger.warn_and_trace(e)
            self.logger.warn("Brand not captured at %s" % url)    
            
        # gender
        try:
            info_json = soup.findAll('span', {'class': re.compile(
                'jsPageContextData')})[-1].get('data-pagecontext')
            gender = json.loads(info_json)['product']['mainCategoryId']
            genderid = self.getGender(gender)
        except:
            gender = ''
            genderid = None
            self.logger.info("Gender not captured at %s" % url)

        # category - sub category
        try:
            info_json = soup.findAll('span', {'class': re.compile(
                    'jsPageContextData')})[-1].get('data-pagecontext')
            # Capture category information    
            prodCatStr = json.loads(info_json)['product']['categoryName']
            # NLP process category
            processedCategories = self.preprocess_metadata(prodCatStr)
            # Get categories from DB and match the captured one
            allCategories = self.db_manager.runSelectQuery(params={'table': 'ProductCategory'})
            matchedCategories = (allCategories.loc[allCategories['Description'].str.lower()
                    .isin( processedCategories.split() ), 'Description'].values)
            if len(matchedCategories>1):
                prodCatID = self.getProductCat(matchedCategories[0])
            else: 
                prodCatID = self.getProductCat(processedCategories)
            prodSubCatID = self.getProductSubCat(prodSubCat=prodCatStr, prodCat=prodCatID)
        except:
            prodCatID = None
            prodSubCatID = None
            self.logger.info("Gender not captured at %s" % url)

        ### Other attributes #
        try:
            attr_list = ['PRODUCT DETAILS:']
            attr_divs = soup.findAll(
                'div', {'class': re.compile('c-productInformation__listContainer')})

            for i, div in enumerate(attr_divs):
                fields = div.findAll(
                    'p', {'class': re.compile('o-copy o-copy--s')})
                for field in fields:
                    if i == 2 and 'MATERIAL & CARE INSTRUCTIONS:' not in attr_list:
                        attr_list.append('MATERIAL & CARE INSTRUCTIONS:')
                        attr_list.append(field.text.replace('\n', '').strip())
                    else:
                        attr_list.append(field.text.replace('\n', '').strip())
            meta = ' - '.join(attr_list)
            meta = self.preprocess_metadata(doc=meta)
        except:
            meta = ''
            self.logger.info("Product Description not captured at %s" % url)
        return price, head, brand, color, genderid, meta, sku, prodCatID, prodSubCatID


    ## Zalando specific functionality 
    ## Fetch search results form Zalando according to 'order' parameter 
    #
    def resultDataframeZalando(self, keyUrl, order, filterDF=pd.DataFrame([]), breakPointNumber=9999999):
        '''
        Fetches search results form Zalando according to 'order' parameter. Parameter "filterDF" is 
        filled with the trending order dataframe to call the script with reference order and get the 
        reference order only for the trending items.
        - Input:
            keyUrl: search URL according to keywords
            order: ordering of results, valid values are 'reference' and 'trend'

        - Returns:
            resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL,
            the product's image URL 
        '''
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None
        # results' ordering
        rule = '&sort=popularity' if ordering == orderDict['reference'] else '&sort=activation_date'
        resultsURL = keyUrl + rule
        
        # Fetch initial search result page
        url, soup = self.get_content(resultsURL)
        jsonInfo = json.loads(str(soup))
        # Capture query error
        if not jsonInfo['articles']:
            self.logger.info('Your search produced no results.')
            return 0

        # Handle pagination and gather all products
        articles = jsonInfo['articles']
        products = articles

        ## Capture product information according to the 'breakPointNumber'
        # if 'breakPointNumber' is less than the 'maxItems' return 'breakPointNumber' items
        # Otherwise parse additional result pages
        maxItems = jsonInfo['total_count']
        breakPointNumber = breakPointNumber if breakPointNumber <= maxItems else maxItems
        offset, limit = len(products), 84
        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL'])
        if order=='reference':  
            breakPointNumber = len(filterDF)   
            while len(products) < maxItems:
                urlList = ['https://www.zalando.co.uk/%s.html' % a['url_key'] for a in articles]
                if not filterDF[filterDF['URL'].isin(urlList)].empty:
                    resultsDF = resultsDF.append(filterDF[filterDF['URL'].isin(urlList)], ignore_index=True)
                if len(resultsDF) >= len(filterDF): # Exit condition for reference order
                    resultsDF = resultsDF[:len(filterDF)]
                    break
                # Prepare next request
                _resultsURL = resultsURL + '&offset=%s' % offset
                url, _soup = self.get_content(_resultsURL)
                _jsonInfo = json.loads(str(_soup))
                articles = _jsonInfo['articles']
                offset += limit
                products += articles
        else:
            maxItems = breakPointNumber 
            while len(products) < maxItems:
                _resultsURL = resultsURL + '&offset=%s' % offset
                url, _soup = self.get_content(_resultsURL)
                _jsonInfo = json.loads(str(_soup))
                articles = _jsonInfo['articles']
                products += articles
                offset += limit      
            products = products[:maxItems]  
            # Iterate result pages
            for product in products:
                # Capture actual products information from json fields
                series = pd.Series([])
                series['Brand'] = product['brand_name']
                series['Head'] = product['name']
                series['SKU'] = product['sku']
                series['Price'] = float(''.join(re.findall(r'[0-9\.]', product['price']['original'])))
                series['URL'] = 'https://www.zalando.co.uk/%s.html' % product['url_key']
                series['imgURL'] = 'https://img01.ztat.net/article/' + product['media'][0]['path']            
                resultsDF = resultsDF.append(series, ignore_index=True)
        # Set ordering                
        resultsDF[ordering] = range(1, len(resultsDF)+1)
        return resultsDF


    # Get product information from Zalando 
    
    def parseZalandoFields(self, url):
        '''
        Gets product information from Zalando
        - Input:
            soup: BeautifulSoup navigator
            url: URL currently parsing

        - Returns:
            head, brand, color, genderid, meta, sku, True, price
        '''
        # Parse product page
        url, soup = self.get_content(url)
        jsonUnparsed = soup .find('script', attrs={'id': re.compile(r'z-vegas-pdp-props')})
        try:
            jsonInfo = re.findall(r'{.+}', str(jsonUnparsed))[0]
            productInfo = json.loads(jsonInfo)
        except Exception as e:        
            self.logger.warn_and_trace(e)
            return {}
        # Color 
        try:
            color = productInfo['model']['articleInfo']['color']
        except Exception as e:
            color = None
            self.logger.info("ColorsDescription not captured at %s" % url)
        # Product category
        try:
            prodCat = productInfo['model']['articleInfo']['silhouette_code']
            prodCatID = self.getProductCat(prodCat)
        except Exception as e:
            prodCat = None
            self.logger.info("ProductCategory not captured at %s" % url)
        # Product subcategory
        try:
            prodSubCat = productInfo['model']['articleInfo']['category_tag']
            prodSubCatID = self.getProductSubCat(prodSubCat=prodSubCat, prodCat=prodCat)
        except Exception as e:
            prodSubCat = None
            self.logger.info("ProductSubCategory not captured at %s" % url)
        # Gender
        try:
            gender = [g for g in ['MEN', 'WOMEN', 'KID', 'MAN', 'WOMAN', 'KIDS', 'UNISEX']  \
                    if g.lower() in map(lambda x: x.lower(), 
                    productInfo['model']['articleInfo']['categories'])][0].lower()
            genderid = self.getGender(gender)
        except Exception as e:
            genderid = None
            self.logger.info("Gender not captured at %s" % url)
        # Metadata
        try:
            attributes = productInfo['model']['articleInfo']['attributes']
            meta_names = [data['name'] for attr in attributes for data in attr['data'] 
                    if attr['category'] in ['heading_material', 'heading_details']]
            meta_values = [data['values'] for attr in attributes for data in attr['data'] 
                    if attr['category'] in ['heading_material', 'heading_details']]
            meta = '. '.join(meta_names + meta_values)
            meta = self.preprocess_metadata(doc=meta)
        except Exception as e:
            meta = ''
            self.logger.info("Metadata not captured at %s" % url)
       
        params = {'ColorsDescription': color, 'Metadata': meta, 'ProductCategory': prodCatID, 
                'ProductSubcategory': prodSubCatID, 'Gender': genderid}
        return params
