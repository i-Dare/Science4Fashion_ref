# -*- coding: utf-8 -*-
########################################################################################################################################################### Import all the libraries needed ###########################################################################################################################################################
import os
import requests
import time
import random
import sqlalchemy
import json
import pandas as pd
import regex as re
from bs4 import BeautifulSoup, ResultSet
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

########### Variables shared amongst all crawlers ###########
CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
# PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config_test.json')
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')
# Get project's configuration file
with open(PROJECT_CONFIG) as f:
    config = json.load(f)
engine = sqlalchemy.create_engine(config['db_connection'] + config['db_name'])


########### Common functionality ###########
########### This function returns the folder name removing the number of images range from they line of keywords file ###########
########### This function will create a soup and returns which is the parsed html format for extracting html tags of the webpage ###########
def get_content(url, suffix=''):
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
    print('\nGet content from: %s' % url+suffix)
    req = requests.get(url+suffix, headers=headers)
    # Retry if blocked
    if req.history:
        print('Request redirected to %s ' % (req.url+suffix))
        new_req = requests.get(req.url+suffix, headers=headers)
        return BeautifulSoup(new_req.content, 'html.parser')
    else:
        return BeautifulSoup(req.content, 'html.parser')


def getFolderName(wholeName):
    tempArray = wholeName.split(" ")
    nameTemp = ""
    for i in range(1, len(tempArray)):
        nameTemp = nameTemp + " " + tempArray[i].strip('"')
    return nameTemp


###########  This function returns the 3rd appearance of / ###########
def hyphen_split(a):
    if a.count("/") == 1:
        return a.split("/")[0]
    else:
        return "/".join(a.split("/", 3)[:3])


########### Convert digital data to binary format ###########
def convertToBinaryData(filename):
    '''
     Convert digital data to binary format
    '''
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


########### Set image destination path ###########
def setImageFilePath(standardUrl, keyword, i):
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
    imageDir = os.path.join(CWD, imgSiteFolder, keyword)
    # Create The folder according to search name
    if not os.path.exists(imageDir):
        os.makedirs(imageDir)
    imageFilePath = os.path.join(imageDir, nameOfFile + ImageType)
    return imageFilePath


########### Get image info according to a valid adentifier ###########
def getImage(imgURL, imageFilePath):
    '''
    Gets image info according to a valid adentifier.
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
    empPhoto = convertToBinaryData(imageFilePath)
    return empPhoto


########### Add new product to the PRODUCT table ###########
def addNewProduct(site, keywords, imageFilePath, empPhoto, url, imgsrc, head, color, genderid, brand, meta):
    '''
    Adds new product info to the PRODUCT table. 
    '''
    print('Adding new product')
    # submit record to the PRODUCT table
    submitdf = pd.DataFrame([{'Crawler': site, 'SearchWords': keywords, 'Image': imageFilePath, 'ImageBlob': empPhoto,
                              'url': url, 'ImageSource': imgsrc, 'SiteClothesHeadline': head, 'Color': color, 'GenderID': genderid, 'Brand': brand,
                              'Metadata': meta, 'ProductCategoryID': None, 'ProductSubcategoryID': None, 'LengthID': None, 'SleeveID': None,
                              'CollarDesignID': None, 'NeckDesignID': None, 'FitID': None, 'ClusterID': -1, 'FClusterID': -1}])
    # submitdf.to_sql('PRODUCT', con=engine, if_exists='append', index=False)
    submitdf.to_sql("PRODUCT", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)


########### Add new product to the PRODUCTHISTORY table ###########
def addNewProductHistory(url, referenceOrder, trendOrder, price):
    '''
    Adds new product info to the PRODUCTHISTORY table.
    '''
    print('Adding new product history')
    # querydf = pd.read_sql_query(
    #     "SELECT * FROM public.\"PRODUCT\" WHERE public.\"PRODUCT\".url = '{}'".format(url), engine)
    querydf = pd.read_sql_query(
        "SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(url), engine)
    prno = querydf['ProductNo'].values[0]
    submitdf = pd.DataFrame(
        [{'ProductNo': prno, 'ReferenceOrder': referenceOrder, 'TrendingOrder': trendOrder, 'Price': price}])
    # submitdf.to_sql("PRODUCTHISTORY", con=engine, if_exists='append', index=False)
    submitdf.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)


########### Get latest product history from PRODUCTHISTORY table ###########
def getRecentHistory(prdno):
    '''
    Gets latest product history from PRODUCTHISTORY table
    '''
    # querydf = pd.read_sql_query(
        # "SELECT * FROM \"PRODUCTHISTORY\" WHERE \"PRODUCTHISTORY\".\"ProductNo\" = '{}'".format(prdno), engine)
    querydf = pd.read_sql_query( "SELECT * FROM SocialMedia.dbo.PRODUCTHISTORY WHERE SocialMedia.dbo.PRODUCTHISTORY.ProductNo = '{}'".format(prdno), engine)

    recentdf = querydf.loc[querydf['SearchDate'] == querydf['SearchDate'].max()].copy()
    # Handle empty values on trending and reference order
    if recentdf['TrendingOrder'].empty:
        recentdf.loc[:, 'TrendingOrder'] = 0
    if recentdf['ReferenceOrder'].empty:
        recentdf.loc[:, 'ReferenceOrder'] = 0
    return recentdf


########### Update product history ###########
def updateProductHistory(prdno, referenceOrder, trendOrder, price, url):
    '''
    Updates product's latest entry in PRODUCTHISTORY table
    '''
    print('Product already exists, updating history')
    # Get info from most recent product record from PRODUCTHISTORY table
    recentdf = getRecentHistory(prdno)
    if not recentdf.empty:
        referenceOrder = recentdf['ReferenceOrder'].values[0]
        # Check if the current values have changed
        if float(recentdf['Price'].values[0]) != price or \
            (recentdf['ReferenceOrder'].values[0] != referenceOrder) or \
            (recentdf['TrendingOrder'].values[0] != trendOrder):
            # The number of images - url  existed in the end of query
            submitdf = pd.DataFrame(
                [{'ProductNo': prdno, 'ReferenceOrder': referenceOrder, 'TrendingOrder': trendOrder, 'Price': price}])
            # submitdf.to_sql("PRODUCTHISTORY", con=engine, if_exists='append', index=False)
            submitdf.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)
    else:
        # Create new entry in PRODUCTHISTORY table
        print('Product %s history not found' % prdno)
        addNewProductHistory(url, referenceOrder, trendOrder, price)


########### Custom functionality, specific to each website ###########
########### Asos specific functionality ###########
########### Fetch search results form Asos according to 'order' parameter ###########
def resultDataframeAsos(keyUrl, order, breakPointNumber=9999999):
    '''
     Fetches search results form s.Oliver according to 'order' parameter.
    - Input:
    keyUrl: search URL according to keywords
    order: ordering of results, valid values are 'reference' and 'trend'

    - Returns:
    resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
    '''
    orderDict = {'reference':'referenceOrder', 'trend':'trendOrder'}
    try:
        ordering = orderDict[order] # results' ordering
    except:
        print('Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
        return None
    parsedItems = 0
    pageNo = 1
    rule = '' if ordering==orderDict['reference'] else '&sort=freshness' # results' ordering
    paging = '&page=%s' % pageNo
    resultsURL = keyUrl + rule +paging

    # Fetch initial search result page
    soup = get_content(resultsURL)
    maxItems = int(soup.find('progress').get('max'))

    ## Prepare ajax request URL 
    # Capture hidden parameters of ajax request for the image loading       
    jsonUnparsed = [script.text for script in soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in script.text][0]
    jsonInfo = json.loads(re.findall(r'JSON\.parse\(\'({\"(?=analytics).+products\":.+\]}})', jsonUnparsed)[0].replace('\\', ''))
    products = jsonInfo['search']['products']

    print('Prepare to fetch results according to %s order' % order)
    # DataFrame to hold results
    resultsDF = pd.DataFrame(columns=[ordering, 'Url','imgURL', 'price'])
    while True:

        for product in products:
            productPage =  product['url']
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
                print("Price not captured for %s" % productPage)  
            
            
            series = pd.Series({'Url': productPage,
                               'imgURL': productImg,
                               'price': price}, 
                               index=resultsDF.columns)
            parsedItems += 1
            resultsDF = resultsDF.append(series, ignore_index=True)
            # Return if number of wanted results is reached
            if len(resultsDF)>breakPointNumber-1:
                resultsDF[ordering] = range(1, len(resultsDF)+1)
                return resultsDF
        # Prepare next result page URL
        pageNo += 1
        paging = '&page=%s' % pageNo
        resultsURL = keyUrl + rule +paging
        # Fetch next search result page
        soup = get_content(resultsURL)

        ## Prepare ajax request URL 
        # Capture hidden parameters of ajax request for the image loading       
        jsonUnparsed = [script.text for script in soup.findAll('script') if 'ANALYTICS_REMOVE_PENDING_EVENT' in script.text][0]
        jsonInfo = json.loads(re.findall('JSON\.parse\(\'({\"(?=analytics).+products\":.+\]}})', jsonUnparsed)[0].replace('\\', ''))
        products = jsonInfo['search']['products']    

        if parsedItems >= maxItems:
            break
    try:
        resultsDF[ordering] = range(1, maxItems+1)
    except:
        print('Failed to fetch eveything. %s/%s (%s%%) fetched' % 
              (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
        resultsDF[ordering] = range(1, len(resultsDF)+1)
    return resultsDF


########### Get product information from s.Oliver ###########
def parseAsosFields(soup, url):
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
        jsonUnparsed = soup.find('script', text = re.compile(r'window\.asos\.pdp\.config\.product.+')).text
        jsonInfo = json.loads(re.findall(r'window\.asos\.pdp\.config\.product = ({.+});', jsonUnparsed)[0])
        ## Get attributes
        # Header
        head = jsonInfo['name']
        # Gender
        gender = jsonInfo['gender']
        if gender.lower() == 'men':
            genderid = 1
        elif gender.lower() == 'women':
            genderid = 2
        # Brand
        brand = jsonInfo['brandName']
        # Color
        color = jsonInfo['variants'][0]['colour']
    except:
        head = None
        genderid = None
        brand = None
        color = None
        print("Header, genderid, brand, and color were not captured at %s" % url)
    
    # Description
    try:
        description = [li.text.strip() for li in soup.find('div', {'class': re.compile('product-description')}).find('ul').findAll('li')]
        about = [soup.find('div', {'class': re.compile('about-me')}).find('p').text.strip()]
        attr_list = description + about
        meta = ' - '.join(attr_list)
    except:
        meta = None
        print("Product Description not captured at %s" % url)        
  
    return head, brand, color, genderid, meta


########### sOliver specific functionality ###########
########### Fetch search results form s.Oliver according to 'order' parameter ###########
def resultDataframeSOliver(keyUrl, order, breakPointNumber=9999999):
    '''
     Fetches search results form s.Oliver according to 'order' parameter.
    - Input:
    keyUrl: search URL according to keywords
    order: ordering of results, valid values are 'reference' and 'trend'
    breakPointNumber: defines the number of results to fetch

    - Returns:
    resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
    '''
    orderDict = {'reference':'referenceOrder', 'trend':'trendOrder'}
    try:
        ordering = orderDict[order] # results' ordering
    except:
        print('Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
        return None
    
    # Fetch initial search result page
    soup = get_content(keyUrl)
    
    # Initialize parameters
    start = 0 # starting point of lazy-loading
    step = 12 # lazy-lading step
    
    ## Prepare ajax request URL 
    # Capture hidden parameters of ajax request for the image lazy-loading
    reqParams = json.loads(soup.find('div', {'class': re.compile("hidden jsOVModel")}).get('data-ovmodel'))
    # Remove irrelevant parameters
    reqParams['params'].pop('qold')
    reqParams['params'].pop('autofilter')

    # Setup custom ajax request
    maxItems = reqParams['hitCount'] # max number of results
    resultRange = 'start=%s&sz=%s' % (start, step) # lazy-loading URL parameter
    rule = 'srule=default&' if ordering==orderDict['reference'] else 'srule=newest-products&' # results' ordering
    suffix = '&lazyloadFollowing=true&view=ajax' # ajx request suffix
    urlAjax = reqParams['urlAjax'] + '?' # initialize ajax request
    # Appeding parameters to custom ajax request
    for k,v in reqParams['params'].items():
        urlAjax += '%s=%s&' % (k, '%20'.join(v.split(' ')))
    urlAjax += rule + resultRange + suffix
    
    print('Prepare to fetch results according to %s order' % order)
    # DataFrame to hold results
    resultsDF = pd.DataFrame(columns=[ordering, 'Url','imgURL'])
    # lazy-loading iteration until reaching the maxItems - 12 products in each iteration
    while True :
        # Fetch results from lazy-loading
        soupAjax = get_content(urlAjax)
        products = soupAjax.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})

        for product in products:
            productPage = product.find('a', {'class': re.compile("js-ovlistview-productdetaillink")}).get('href')
            productPage = 'https://www.soliver.eu' + productPage
            productImg = json.loads(product.findAll('div', {'class': re.compile("lazyload jsLazyLoad")})[0].get('data-picture'))['sources'][0]['srcset'].split(',')[0]
            series = pd.Series({'Url': productPage.rsplit('?', 1)[0],
                               'imgURL':productImg}, 
                               index=resultsDF.columns)
            resultsDF = resultsDF.append(series, ignore_index=True)
            # Return if number of wanted results is reached
            if len(resultsDF)>breakPointNumber-1:
                resultsDF[ordering] = range(1, len(resultsDF)+1)
                return resultsDF
            
        if start + step >= maxItems:
            break
            
        # Prepare next ajax request
        start+=step
        resultRange = 'start=%s&sz=%s' % (start, step)        
        urlAjax = re.sub(r'start=[0-9]+&sz=[0-9]+', resultRange, urlAjax)
            
    try:
        resultsDF[ordering] = range(1, maxItems+1)
    except:
        print('Failed to fetch eveything. %s/%s (%s%%) fetched' % 
              (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
        resultsDF[ordering] = range(1, len(resultsDF)+1)
    return resultsDF


########### Get product information from s.Oliver ###########
def parseSOliverFields(soup, url, imgURL):
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
    pid = ''
    for hidden in soup.findAll('div', {'class': re.compile('hidden')}):
        if hidden.get('data-mms-spv-variationmodel'):
            jsonInfoImg = json.loads(hidden.get('data-mms-spv-variationmodel'))
            for swatch in jsonInfoImg['colorSwatches']:
                if swatch['swatchImage']['img']['src'].split('_')[0] in imgURL:
                    pid = swatch['variantID']
                    break
            break
    
    ajaxURL = 'https://www.soliver.eu/on/demandware.store/Sites-soliverEU-Site/en/Catalog-GetProductData?pview=variant&pid=' 
    ajaxURL += pid
    # return ajax request for the selected image
    soupTemp = get_content(ajaxURL)
    jsonInfo = json.loads(soupTemp.text)['product']['variantInfo']
    # color - Clothe's Color
    try:
        color = jsonInfo['variationAttrData']['color']['displayValue'].split(' ')[0]
    except:
        color = None
        print("Color not captured at %s" % url)
        
    #price
    try:
        price = jsonInfo['priceData']['listPrice']['value']
        sales_price = jsonInfo['priceData']['salePrice']['value']
        isSale = jsonInfo['priceData']['isSale']
        price = sales_price if isSale else price
    except:
        price = None
        print("Price not captured at %s" % url)
    
    # head - Clothe's General Description
    try:    
        head = jsonInfo['microdata']['description']
    except:
        head = None
        print("Header not captured at %s" % url)
    
    # brand - Clothe's Brand
    try:
        brand = jsonInfo['microdata']['brand']
    except:
        brand = None
        print("Brand not captured at %s" % url)
        
    # gender
    try:
        info_json = soup.findAll('span', {'class': re.compile('jsPageContextData')})[-1].get('data-pagecontext')
        gender = json.loads(info_json)['product']['mainCategoryId']
        if gender.lower() == 'men':
            genderid = 1
        elif gender.lower() == 'women':
            genderid = 2
        elif gender.lower() == 'junior':
            genderid = 3
    except:
        gender = ''
        genderid = None
        print("Gender not captured at %s" % url)

    ############ Other attributes ############
    try:
        attr_list = ['PRODUCT DETAILS:']
        attr_divs = soup.findAll('div', {'class': re.compile('c-productInformation__listContainer')})

        for i,div in enumerate(attr_divs):
            fields = div.findAll('p', {'class': re.compile('o-copy o-copy--s')})
            for field in fields:
                if i==2 and 'MATERIAL & CARE INSTRUCTIONS:' not in attr_list:
                    attr_list.append('MATERIAL & CARE INSTRUCTIONS:')
                    attr_list.append(field.text.replace('\n', ' ').strip())
                else:
                    attr_list.append(field.text.replace('\n', ' ').strip())
        meta = ' - '.join(attr_list)
    except:
        meta = None
        print("Product Description not captured at %s" % url)
    return price, head, brand, color, genderid, meta


########### Zalando specific functionality ###########
########### Fetch search results form Zalando according to 'order' parameter ###########
def resultDataframeZalando(keyUrl, order, breakPointNumber=9999999):
    '''
     Fetches search results form s.Oliver according to 'order' parameter.
    - Input:
    keyUrl: search URL according to keywords
    order: ordering of results, valid values are 'reference' and 'trend'

    - Returns:
    resultsDF: pandas DataFrame with columns the ordering sequence, the product webpage URL, the product's image URL 
    '''
    orderDict = {'reference':'referenceOrder', 'trend':'trendOrder'}
    try:
        ordering = orderDict[order] # results' ordering
    except:
        print('Result parser error: \"order\" argument is either \"reference\" or \"trend\"')
        return None
    pageNo = 1
    rule = '&p=%s' % pageNo if ordering==orderDict['reference'] else '&p=%s&order=activation_date' % pageNo # results' ordering
    resultsURL = keyUrl + rule
    
    # Fetch initial search result page
    soup = get_content(resultsURL)
    
    ## Prepare ajax request URL 
    # Capture hidden parameters of request
    reqParams = json.loads(soup.find('script', {'id': re.compile("z-nvg-cognac-props")}).text[9:-3])
    maxItems = reqParams['total_count']
    max_pageNo = reqParams['pagination']['page_count']
    
    # DataFrame to hold results
    resultsDF = pd.DataFrame(columns=[ordering, 'Url','imgURL'])
    # Iterate result pages    
    while pageNo <= max_pageNo:
        # Capture products in the page
        products = json.loads(soup.find('script', {'id': re.compile("z-nvg-cognac-props")}).text[9:-3])['articles']
        for product in products:
            productPage = 'https://www.zalando.co.uk/%s.html' % product['url_key']
            productImg = 'https://img01.ztat.net/article/' + product['media'][0]['path']
            series = pd.Series({'Url': productPage.rsplit('?', 1)[0],
                                'imgURL':productImg}, 
                                index=resultsDF.columns)
            # Update trend DateFrame
            resultsDF = resultsDF.append(series, ignore_index=True)
            if len(resultsDF)>breakPointNumber-1:
                resultsDF[ordering] = range(1, len(resultsDF)+1)
                return resultsDF

        # Prepare next request
        pageNo += 1
        resultsURL = re.sub(r'&p=[0-9]+', '&p=%s' % pageNo, resultsURL)
        soup = get_content(resultsURL)
    try:
        resultsDF[ordering] = range(1, maxItems+1)
    except:
        print('Failed to fetch eveything. %s/%s (%s%%) fetched' % 
              (len(resultsDF), maxItems, round(len(resultsDF)*100/maxItems, 2)))
        resultsDF[ordering] = range(1, len(resultsDF)+1)
    return resultsDF


########### Get product information from Zalando ###########
def parseZalandoFields(soup, url):
    '''
    Gets product information from Zalando
    - Input:
    soup: BeautifulSoup navigator
    url: URL currently parsing

    - Returns:
    price, head, brand, color, genderid, meta
    '''
    # price
    flagprice = False
    try:
        price = soup.find('span', text=re.compile(
            r'[0-9]+[\.,][0-9]')).text.replace(',', '.')
        price = float(re.findall(r"[+-]?\d+\.\d+", price)[0])
    except:
        flagprice = True
        price = None
        print("Price not captured at %s" % url)

    # head - Clothe's General Description
    try:
        head = soup.find('h1').text
        head = re.sub('\s+', ' ', head).strip()
    except:
        head = ''
        print("Header not captured at %s" % url)
    # brand - Clothe's Brand
    try:
        brand = soup.find('h3').text
        brand = re.sub('\s+', ' ', brand).strip()
    except:
        brand = ''
        print("Brand not captured at %s" % url)
    # color - Clothe's Color
    try:
        color = soup.findAll(text=re.compile(r'Colour'))[-1].find_next().text
    except:
        color = ''
        print("Color not captured at %s" % url)
    # gender - Clothe's Gender
    try:
        info_json = soup.find(text=re.compile(r'navigationTargetGroup'))
        gender = json.loads(info_json)[
            'rootEntityData']['navigationTargetGroup']
        if gender.lower() == 'men':
            genderid = 1
        elif gender.lower() == 'women':
            genderid = 2
        else:
            genderid = 3
    except:
        gender = ''
        genderid = None
        print("Gender not captured at %s" % url)
    # Other attributes
    meta = ''
    # For Description
    try:
     # parse metadata
        parse_info = soup.find(text=re.compile(r'heading_details'))
        json_info = parse_info[parse_info.find('{'):len(parse_info)-parse_info[::-1].find('}')]
        attr_list = []
        for field in json.loads(json_info)['model']['productDetailsCluster']:
            for data in field['data']:
                if 'values' in data.keys():
                    attr_list.append(data['values'])
        # keep only unique attributes
        attr_list = list(set(attr_list))
        meta = ' - '.join(attr_list)
    except:
        print("Product Description not captured at %s" % url)
    return price, head, brand, color, genderid, meta

