# -*- coding: utf-8 -*-
########################################### Import all the libraries needed ###########################################
import os
import json
import time
import requests
import sqlalchemy
from core.helper_functions import *
import pandas as pd
import regex as re

from bs4 import BeautifulSoup, ResultSet
from datetime import datetime
import core.config as config


############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ############
def performScraping(urlReceived, category):
   # initialize scraping
    start_time = time.time()
    # stats counters
    numUrlExist = 0
    numImagesDown = 0

    # Image download directory
    folderIndividualName = os.path.join(currendDir, 'temp')
    if not os.path.exists(folderIndividualName):
        os.makedirs(folderIndividualName)

    url, soup = helper_functions.get_content(urlReceived)
    # Get all relevant results for searh term
    refDF = helper_functions.crawlingSOliver(urlReceived, 'reference')
    # Get trending products
    trendDF = helper_functions.crawlingSOliver(urlReceived, 'trend')

    # Iterate trending products
    for i, row in trendDF.iterrows():
        trendOrder = row['trendOrder']
        url = row['URL']
        imgURL = row['imgURL']
        # Retrieve reference order
        if not refDF.loc[refDF['URL']==url, 'referenceOrder'].empty:
            referenceOrder = refDF.loc[refDF['URL']==url, 'referenceOrder'].values[0]
        else:
            referenceOrder = 0

        # Find fields from product's webpage
        url, soup = helper_functions.get_content(url)
        price, head, brand, color, genderID, meta, sku, isActive = helper_functions.parseSOliverFields(soup, url, imgURL)

        url=url.replace("'", "''")
        querydf = pd.read_sql_query("SELECT * FROM %s.dbo.Product WHERE  CONVERT(VARCHAR(MAX), %s.dbo.PRODUCT.URL) = \
            CONVERT(VARCHAR(MAX),'%s')" % (dbName, dbName, str(url)), engine)
        # querydf = pd.read_sql_query("SELECT * FROM public.\"Product\" WHERE public.\"Product\".\"URL\" = '%s'" %  url.replace("%", "%%"), engine)
        if not querydf.empty:
            # Update ProductHistory
            prdno = querydf['Oid'].values[0]
            helper_functions., logfile=logfile).logger(prdno, referenceOrder, trendOrder, price, url, '')
            numUrlExist += 1
            print('Info for product %s updated' % prdno)
        else:
            # Download product image            
            dt_string = datetime.now().strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
            nameOfFile = 'image' + str(i) + "_" + re.sub("[ /:]", ".", dt_string)
            ImageType = ".jpeg"
            imageFilePath = os.path.join(currendDir, 'temp', nameOfFile + ImageType)
            empPhoto = helper_functions.getImage(imgURL, imageFilePath)
            os.remove(imageFilePath)
            numImagesDown += 1
            print('Image number %s: %s' % (trendOrder, imageFilePath.split(os.sep)[-1]))

             # Create new entry in PRODUCT table
            helper_functions.addNewProduct(
                site, category, imageFilePath, empPhoto, url, imgURL, head, color, genderID, brand, meta, sku, isActive)

            # Create new entry in ProductHistory table
            helper_functions.addNewProductHistory(url, referenceOrder, trendOrder, price, '')

    os.rmdir(folderIndividualName)
    print('Images requested: %s,   Images Downloaded: %s (%s%%),   Images Existed: %s' % (len(trendDF), numImagesDown, round(numImagesDown/len(trendDF),2 ) * 100, numUrlExist))
    # The time needed to scrape this query
    print("Time to scrape category %s: %s seconds ---" % (category, round(time.time() - start_time, 2)))


############ Main function ############
if __name__ == '__main__':
    start_time_all = time.time()
    
    currendDir = config.WEB_CRAWLERS
    engine = config.ENGINE
    dbName = config.DB_NAME
    # Webpage URL
    standardUrl = 'https://www.soliver.eu/'    
    site = str((standardUrl.split('.')[1]).capitalize())
    # Capture categories and category URLs form header
    url, soup = helper_functions.get_content(standardUrl)    
    category, categoryURLs = [], []
    jsonInfo = json.loads(soup.findAll('span', {'data-pagecontext': re.compile('showAllProducts')})[0].get('data-pagecontext'))
    for element in jsonInfo['global']['navigationTree']:
        for child in element['childs']:
            if element['name'].lower() in ['men', 'women', 'man', 'woman'] and re.match('cloth.+', child['name'], re.IGNORECASE):
                category.append(child['name'])
                categoryURLs.append(standardUrl + child['url'])
            elif child['name'].lower() in ['boys', 'girls', 'baby']:
                for c in child['childs']:
                    category.append(child['name'])
                    categoryURLs.append(standardUrl + c['url'])
            
    for cat,url in zip(category, categoryURLs):
        print('Category: %s, URL: %s' % (cat, url))
        performScraping(url, cat)

    print("Time to scrape ALL queries is %s seconds ---" % (time.time() - start_time_all))
