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
import wordsegment
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
    def get_content(self, url, suffix=''):
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
        self.logger.info('Get content from: %s' % url+suffix)
        try:
            req = requests.get(url+suffix, headers=headers, timeout=config.CRAWLER_TIMEOUT ,verify=False)
            soup = BeautifulSoup(req.content, 'html.parser')
        except requests.exceptions.Timeout:
            self.logger.warning("Timeout occurred when getting %s" % url+suffix)
            return None
        new_req = ''
        if req.history:
            self.logger.info('Request redirected to %s ' % (req.url+suffix))
            try:
                new_req = requests.get(req.url+suffix, headers=headers, timeout =10 ,verify=False)
                soup = BeautifulSoup(new_req.content, 'html.parser')
            except requests.exceptions.Timeout:
                self.logger.warning("Timeout occurred when getting %s" % url+suffix)     
                return None  

        return soup
        

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


    ## Convert blob to img
    #
    def convertBlobToImage(self, blob):        
        x = np.frombuffer(blob, dtype='uint8')
        # decode the array into an image
        img = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
        return img


    def convertCVtoPIL(self, img):
        ###Convert image fron OpenCV to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(img)
        return im_pil    


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



    def getImage(self, imgURL, imageFilePath):
        '''
        Gets image from a URL and saves it to given file path
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

    
# --------------------------------------------------------------------------                        
#          Database IO Functionality
# -------------------------------------------------------------------------- 

    ## Return gender ID according to the extracted text from the crawler
    #
    def getGender(self, gender):
        gender_dict = {
            'man': ['man', 'men', 'male'],
            'woman': ['woman', 'women', 'female'],
            'kids': ['kid', 'kids', 'child', 'children'],
            'unisex': ['unisex']}

        genderdf = pd.read_sql_query("SELECT * FROM %s.dbo.Gender" % config.DB_NAME, config.ENGINE)
        genderdf_dict = genderdf.set_index('Oid').loc[:, 'Description'].str.lower().to_dict()

        # Find gender in gender_dict
        gnd = [g_label for g_label, g in gender_dict.items() if gender.lower() in g][0]
        # Find captured gender in gender dataframe
        gender_id = [g_id for g_id, g in genderdf_dict.items() if gnd==g][0]
        return gender_id   


    ## Add new product to the Product table 
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
        

    ## Add new product to the Product table 
    #
    def addNewProduct(self, site, keywords, imageFilePath, empPhoto, url, imgsrc, head, color, 
                            genderid, brand, meta, sku, isActive, price=None):
        '''
        Adds new product info to the Product table. 
        '''
        queryAdaptersdf = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter WHERE \
                %s.dbo.Adapter.Description = '%s'" % (config.DB_NAME, config.DB_NAME, site), config.ENGINE)
        # Add a new Brand if it does not exist dn return the 'Oid' or just return existing 'Oid'
        brandID = self.getBrand(brand)        
        
        ## Add product and return new records in DataFrame
        # Check if product exists
        # If DataFrame is empty add new product
        self.logger.info('Adding new product')
        uniq_params = {'table': 'Product', 'URL': url}
        params = {'table': 'Product', 'Adapter': queryAdaptersdf.loc[0, 'Oid'], 'Description': keywords, 
                'AlternativeDescription': None, 'Active':  True, 'Ordering': 0, 'ProductCode': sku, 
                'ProductTitle': head, 'Composition': None, 'ForeignComposition': None, 'SiteHeadline': head, 
                'ColorsDescription': color, 'Metadata': meta, 'SamplePrice': None, 'ProductionPrice': None, 
                'WholesalePrice': None, 'RetailPrice': price, 'Image': empPhoto, 'Photo': imageFilePath, 
                'Sketch': None, 'URL': url, 'ImageSource': imgsrc, 'Brand': brandID, 'Fit': None, 
                'CollarDesign': None, 'SampleManufacturer': None, 'ProductionManufacturer': None, 
                'Length': None, 'NeckDesign': None, 'ProductCategory': None, 'ProductSubcategory': None, 
                'Sleeve': None, 'LifeStage': None, 'TrendTheme': None, 'InspirationBackground': None, 
                'Gender': genderid, 'BusinessUnit': None, 'Season': None, 'Cluster': None, 'FinancialCluster': None, 
                'OptimisticLockField': None}
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
#          Natural Language Processing Functionality 
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
        doc = re.sub('[' + re.escape(string.punctuation) + ']+', ' ', doc)
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

# --------------------------------------------------------------------------                        
#          Web crawler functionality, specific for each website
# --------------------------------------------------------------------------
    ## Generic functionaloty
    # Add/update product indormation
    def registerData(self, site, standardUrl, url, imgURL, searchTerm, referenceOrder, trendOrder, 
                     cnt, crawlFunc):                     
        # Check if product url exists to decide if addition of update is needed
        params = {'table': 'Product', 'URL': url}
        product_df = self.db_manager.runSelectQuery(params)

        if not product_df.empty:
            action = 'update' # action flag
            ## Product exists, proceed to update ProductHistory table
            # Update ProductHistory
            _df = self.updateProductHistory(product_df, referenceOrder, trendOrder)
            cnt += 1
            
        else:
            action = 'addition' # action flag
            # Download product image
            imageFilePath = self.setImageFilePath(standardUrl, ''.join(searchTerm.split()), trendOrder)
            empPhoto = self.getImage(imgURL, imageFilePath)
            cnt += 1
            # Find fields from product's webpage
            soup = self.get_content(url)
            head, brand, color, genderid, meta, sku, isActive, price = crawlFunc(soup, url)
            # Create new entry in PRODUCT table
            product_df = self.addNewProduct(site, searchTerm, imageFilePath, empPhoto, url, 
                                                imgURL, head, color, genderid, brand, meta, sku, 
                                                isActive, price)

            # Create new entry in ProductHistory table
            _df = self.addNewProductHistory(product_df, referenceOrder, trendOrder)
        productID = _df.loc[0, 'Product']
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
        start_time = time.time()
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None
        parsedItems = 0
        pageNo = 1
        # results' ordering
        rule = '' if ordering == orderDict['reference'] else '&sort=freshness'
        paging = '&page=%s' % pageNo
        resultsURL = keyUrl + rule + paging

        # Fetch initial search result page
        soup = self.get_content(resultsURL)
        maxItems = int(soup.find('progress').get('max'))

        # Prepare ajax request URL
        # Capture hidden parameters of ajax request for the image loading
        try:
            jsonUnparsed = [script for script in soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in str(script)]
            jsonInfo = json.loads(re.findall(r'JSON\.parse.+({\"(?=analytics).+products\".+\]}}})', str(jsonUnparsed))[0].replace('\\', ''))
            products = jsonInfo['search']['products']
            breakPointNumber = len(products)
        except Exception as e:        
            self.logger.warn_and_trace(e)

        self.logger.info('Prepare to fetch results according to %s order' % order)
        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL', 'price'])
        while True:

            for product in products:
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

                series = pd.Series({'URL': productPage.replace("'", "''").replace("%", "%%"),
                                    'imgURL': productImg,
                                    'price': price},
                                index=resultsDF.columns)            
                
                # Return if number of wanted results is reached
                if not filterDF.empty :
                    if series['URL'] in filterDF['URL'].values:
                        resultsDF = resultsDF.append(series, ignore_index=True)
                        parsedItems += 1
                        if len(resultsDF) >= len(filterDF):
                            resultsDF[ordering] = range(1, len(resultsDF)+1)
                            return resultsDF
                else:
                    resultsDF = resultsDF.append(series, ignore_index=True)
                    parsedItems += 1
                    if len(resultsDF) > breakPointNumber-1:
                        resultsDF[ordering] = range(1, len(resultsDF)+1)
                        return resultsDF

            # Prepare next result page URL
            pageNo += 1
            paging = '&page=%s' % pageNo
            resultsURL = keyUrl + rule + paging
            # Fetch next search result page
            soup = self.get_content(resultsURL)

            # Prepare ajax request URL
            # Capture hidden parameters of ajax request for the image loading
            jsonUnparsed = [script.text for script in soup.findAll(
                'script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in script.text][0]
            jsonInfo = json.loads(re.findall(
                r'JSON\.parse\(\'({\"(?=analytics).+products\":.+\]}})', jsonUnparsed)[0].replace('\\', ''))
            products = jsonInfo['search']['products']

            if parsedItems >= maxItems or len(resultsDF) > breakPointNumber-1:
                break
        try:
            resultsDF[ordering] = range(1, maxItems+1)
        except:
            self.logger.info('Failed to fetch eveything. %s/%s (%s%%) fetched' %
                (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
            resultsDF[ordering] = range(1, len(resultsDF)+1)
        
        self.logger.info("Time to retrieve %s results: %s seconds ---" % (order, round(time.time() - start_time, 2)))
        return resultsDF


    ## Get product information from Asos 
    #
    def parseAsosFields(self, soup, url):
        '''
        Gets product information from Zalando
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
        except Exception as e: 
            meta = None
            self.logger.warn_and_trace(e)
            self.logger.warning("Product Description not captured at %s" % url)
        # Price
        try:
            productID = jsonInfo['id']
            priceApiURL = 'https://www.asos.com/api/product/catalogue/v3/stockprice?productIds=%s&store=ROE&currency=EUR' % productID
            time.sleep(3)  # suspend execution for 5 secs
            priceSoup = self.get_content(priceApiURL)
            price = json.loads(str(priceSoup))[0]['productPrice']['current']['value']
        except:
            price = None
            self.logger.warning("Product price not captured at %s" % url)   
    

        return head, brand, color, genderid, meta, sku, True, price


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
        start_time = time.time()
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None

        # Fetch initial search result page
        soup = self.get_content(keyUrl)

        # Initialize parameters
        start = 0  # starting point of lazy-loading
        step = 12  # lazy-lading step

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
        resultRange = 'start=%s&sz=%s' % (
            start, step)  # lazy-loading URL parameter
        # results' ordering
        rule = 'srule=default&' if ordering == orderDict['reference'] else 'srule=newest-products&'
        suffix = '&lazyloadFollowing=true&view=ajax'  # ajx request suffix
        urlAjax = reqParams['urlAjax'] + '?'  # initialize ajax request
        # Appeding parameters to custom ajax request
        for k, v in reqParams['params'].items():
            urlAjax += '%s=%s&' % (k, '%20'.join(v.split(' ')))
        urlAjax += rule + resultRange + suffix

        self.logger.info('Prepare to fetch results according to %s order' % order)
        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL'])
        # lazy-loading iteration until reaching the maxItems - 12 products in each iteration
        while True:
            # Fetch results from lazy-loading
            soupAjax = self.get_content(urlAjax)
            products = soupAjax.findAll(
                'div', {'class': re.compile("productlist__product js-ovgrid-item")})

            for product in products:
                productPage = product.find('a', {'class': re.compile(
                    "js-ovlistview-productdetaillink")}).get('href')
                productPage = 'https://www.soliver.eu' + productPage
                productImg = json.loads(product.findAll('div', {'class': re.compile("lazyload jsLazyLoad")})[
                                        0].get('data-picture'))['sources'][0]['srcset'].split(',')[0]
                series = pd.Series({'URL': productPage.rsplit('?', 1)[0].replace("'", "''").replace("%", "%%"),
                                    'imgURL': productImg},
                                index=resultsDF.columns)
                
                # Return if number of wanted results is reached
                if not filterDF.empty :
                    if series['URL'] in filterDF['URL'].values:
                        resultsDF = resultsDF.append(series, ignore_index=True)
                        if len(resultsDF) >= len(filterDF):
                            resultsDF[ordering] = range(1, len(resultsDF)+1)
                            return resultsDF
                else:
                    resultsDF = resultsDF.append(series, ignore_index=True)
                    if len(resultsDF) > breakPointNumber-1:
                        resultsDF[ordering] = range(1, len(resultsDF)+1)
                        return resultsDF

            if start + step >= maxItems:
                break

            # Prepare next ajax request
            start += step
            resultRange = 'start=%s&sz=%s' % (start, step)
            urlAjax = re.sub(r'start=[0-9]+&sz=[0-9]+', resultRange, urlAjax)

        try:
            resultsDF[ordering] = range(1, maxItems+1)
        except:
            self.logger.info('Failed to fetch eveything. %s/%s (%s%%) fetched' %
                (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
            resultsDF[ordering] = range(1, len(resultsDF)+1)
        self.logger.info("Time to retrieve %s results: %s seconds ---" % (order, round(time.time() - start_time, 2)))
        return resultsDF


    ## Get product information from s.Oliver 
    #
    def parseSOliverFields(self, soup, url, imgURL):
        '''
        Gets product information from Zalando
        - Input:
            soup: BeautifulSoup navigator
            url: URL currently parsing
            imgURL: image URL

        - Returns:
            price, head, brand, color, genderid, meta
        '''
        # Get product id according to the captured image URL
        pid, sku, isActive = '', '', False
        for hidden in soup.findAll('div', {'class': re.compile('hidden')}):
            if hidden.get('data-mms-spv-variationmodel'):
                jsonInfoImg = json.loads(hidden.get('data-mms-spv-variationmodel'))
                for swatch in jsonInfoImg['colorSwatches']:
                    if swatch['swatchImage']['img']['src'].split('_')[0] in imgURL:
                        pid = swatch['variantID']
                        sku = pid
                        break
                for colorSize in jsonInfoImg['availableColorSizes']:
                    for size in jsonInfoImg['availableColorSizes'][colorSize]:
                        if jsonInfoImg['availableColorSizes'][colorSize][size]['variantID'] == sku:         
                            isActive = jsonInfoImg['availableColorSizes'][colorSize][size]['isAvailable']
                            break
                
        if sku == '':
            self.logger.info("ProductCode and active information not captured at %s" % url)
        ajaxURL = 'https://www.soliver.eu/on/demandware.store/Sites-soliverEU-Site/en/Catalog-GetProductData?pview=variant&pid='
        ajaxURL += pid
        # return ajax request for the selected image
        soupTemp = self.get_content(ajaxURL)
        try:
            jsonInfo = json.loads(soupTemp.text)['product']['variantInfo']
        except:
            self.logger.info("JSON information for product at %s not found" % url)
        # color - Clothe's Color
        try:
            color = jsonInfo['variationAttrData']['color']['displayValue']
        except:
            color = None
            self.logger.info("Color not captured at %s" % url)

        # price
        try:
            price = jsonInfo['priceData']['salePrice']['value']
        except:
            price = None
            self.logger.info("Price not captured at %s" % url)

        # head - Clothe's General Description
        try:
            head = jsonInfo['microdata']['description']
        except:
            head = None
            self.logger.info("Header not captured at %s" % url)

        # brand - Clothe's Brand
        try:
            brand = jsonInfo['microdata']['brand']
        except:
            brand = None
            self.logger.info("Brand not captured at %s" % url)    
            
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
                        attr_list.append(field.text.replace('', ' ').strip())
                    else:
                        attr_list.append(field.text.replace('', ' ').strip())
            meta = ' - '.join(attr_list)
        except:
            meta = None
            self.logger.info("Product Description not captured at %s" % url)
        return price, head, brand, color, genderid, meta, sku, isActive


    ## Zalando specific functionality 
    ## Fetch search results form Zalando according to 'order' parameter 
    #
    def resultDataframeZalando(self, keyUrl, order, filterDF=pd.DataFrame([]), breakPointNumber=9999999):
        '''
        Fetches search results form Zalando according to 'order' parameter. Parameter "filterDF" is 
        filled with the trending order dataframe to call the script with reference order and get the reference order
        only for the trending items.
        - Input:
            keyUrl: search URL according to keywords
            order: ordering of results, valid values are 'reference' and 'trend'

        - Returns:
            resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
        '''
        start_time = time.time()
        orderDict = {'reference': 'referenceOrder', 'trend': 'trendOrder'}
        try:
            ordering = orderDict[order]  # results' ordering
        except:
            self.logger.info(
                'Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
            return None
        pageNo = 1
        # results' ordering
        rule = '&p=%s' % pageNo if ordering == orderDict[
            'reference'] else '&p=%s&order=activation_date' % pageNo
        resultsURL = keyUrl + rule

        # Fetch initial search result page
        soup = self.get_content(resultsURL)

        # Prepare ajax request URL
        # Capture hidden parameters of request
        reqParams = json.loads(
            str(soup.findAll('script', {'id': re.compile("z-nvg-cognac-props")})[0].next)[9:-3])
        maxItems = reqParams['total_count']
        max_pageNo = reqParams['pagination']['page_count']

        # DataFrame to hold results
        resultsDF = pd.DataFrame(columns=[ordering, 'URL', 'imgURL'])
        # Iterate result pages
        while pageNo <= max_pageNo:
            # Capture products in the page
            products = json.loads(
                    str(soup.findAll('script', {'id': re.compile("z-nvg-cognac-props")})[0].next)[9:-3])['articles']
            for product in products:
                productPage = 'https://www.zalando.co.uk/%s.html' % product['url_key']
                productImg = 'https://img01.ztat.net/article/' + \
                    product['media'][0]['path']
                series = pd.Series({'URL': productPage.rsplit('?', 1)[0].replace("'", "''").replace("%", "%%"),
                                    'imgURL': productImg},
                                index=resultsDF.columns)
                # Return if number of wanted results is reached
                if not filterDF.empty :
                    if series['URL'] in filterDF['URL'].values:
                        resultsDF = resultsDF.append(series, ignore_index=True)
                        if len(resultsDF) >= len(filterDF):
                            resultsDF[ordering] = range(1, len(resultsDF)+1)
                            return resultsDF
                else:
                    resultsDF = resultsDF.append(series, ignore_index=True)
                    if len(resultsDF) > breakPointNumber-1:
                        resultsDF[ordering] = range(1, len(resultsDF)+1)
                        return resultsDF

            # Prepare next request
            pageNo += 1
            resultsURL = re.sub(r'&p=[0-9]+', '&p=%s' % pageNo, resultsURL)
            soup = self.get_content(resultsURL)
        try:
            resultsDF[ordering] = range(1, maxItems+1)
        except:
            self.logger.info('Failed to fetch eveything. %s/%s (%s%%) fetched' %
                (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
            resultsDF[ordering] = range(1, len(resultsDF)+1)
        self.logger.info("Time to retrieve %s results: %s seconds ---" % (order, round(time.time() - start_time, 2)))
        return resultsDF


    ## Get product information from Zalando 
    #
    def parseZalandoFields(self, soup, url, imgURL):
        '''
        Gets product information from Zalando
        - Input:
            soup: BeautifulSoup navigator
            url: URL currently parsing

        - Returns:
            price, head, brand, color, genderid, meta
        '''
        # price
        try:
            price = soup.find('span', text=re.compile(
                r'[0-9]+[\.,][0-9]')).text.replace(',', '.')
            price = float(re.findall(r"[+-]?\d+\.\d+", price)[0])
        except:
            price = None
            self.logger.info("Price not captured at %s" % url)

        # head - Clothe's General Description
        try:
            head = soup.find('h1').text
            head = re.sub(r'\s+', ' ', head).strip()
        except:
            head = ''
            self.logger.info("Header not captured at %s" % url)
            
        # color, brand, sku, active information
        sku = ''
        color = ''   
        brand = '' 
        img = imgURL.split('?')[0]
        try:
            parseInfo = soup.find('script', attrs={'id': re.compile(r'z-vegas-pdp-props')}).text
            jsonInfo = json.loads(parseInfo[parseInfo.find('{'):len(parseInfo)-parseInfo[::-1].find('}')])
            items = jsonInfo['model']['articleInfo']['colors'] 
        except Exception as e: 
            self.logger.info(e)
            self.logger.info("Failed to parse jsonInfo at  %s" % url)
        try:    
            for i, item in enumerate(items):
                for imgColor in item['media']['images']:
                    if img in imgColor['sources']['color']:
                        color = items[i]['color']
                        sku = items[i]['id']
                        break
            isActive = jsonInfo['model']['articleInfo']['active']
            brand = jsonInfo['model']['articleInfo']['brand']['name']
        except:
            color = None
            sku = None
            isActive = 1
            self.logger.info("Color, brand, sku, active not captured at %s" % url)
        # gender - Clothe's Gender
        try:
            jsonInfo = soup.find(text=re.compile(r'navigationTargetGroup'))
            gender = json.loads(jsonInfo)[
                'rootEntityData']['navigationTargetGroup']
            genderid = self.getGender(gender)
        except:
            gender = ''
            genderid = None
            self.logger.info("Gender not captured at %s" % url)
        # Other attributes
        # For Description
        meta = ''
        try:
            # parse metadata
            parseInfo = soup.find(text=re.compile(r'heading_details'))
            jsonInfo = parseInfo[parseInfo.find('{'):len(parseInfo)-parseInfo[::-1].find('}')]
            attr_list = []
            for field in json.loads(jsonInfo)['model']['productDetailsCluster']:
                for data in field['data']:
                    if 'values' in data.keys():
                        attr_list.append(data['values'])
            # keep only unique attributes
            attr_list = list(set(attr_list))
            meta = ' - '.join(attr_list)
        except:
            self.logger.info("Product Description not captured at %s" % url)
        return price, head, brand, color, genderid, meta, sku, isActive