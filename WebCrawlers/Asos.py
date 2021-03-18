# -*- coding: utf-8 -*-
############ Import all the libraries needed ############
import os
import json
import time
import requests
import sqlalchemy
import pandas as pd
import time
import regex as re

from bs4 import BeautifulSoup, ResultSet
from datetime import datetime
import sys
from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger


############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the searchTerm text file ############
def performScraping(urlReceived, searchTerm, breakPointNumber):
    # stats counters
    numUrlExist = 0
    numImagesDown = 0
    ## Check no results page
    # Get search results page
    soup = helper.get_content(urlReceived)
    
    # If timeout or no result is returned
    try:
        noResultsMessage = soup.find('link', {'href': re.compile(r'no-results-page')})
        if noResultsMessage:
            logger.info('Unfortunately, your search for produced no results.')
            return 0
    except:
        return None

    # If soup is not redirected to a search page
    #    parse single product
    # else parse according to reference and trending order
    if 'Search' not in soup.title :
        logger.warning('Redirected to product page')
        jsonUnparsed = soup.find('script', {'id': re.compile(r'split-structured-data')})
        product = json.loads(re.findall(r'({\".+})', str(jsonUnparsed))[0].replace('\\', ''))
        breakPointNumber = 1
        url = product['url']        
        imgURL = product['image']
        #  Download product image
        trendOrder = referenceOrder = 0
        imageFilePath = helper.setImageFilePath(standardUrl, ''.join(searchTerm.split()), trendOrder)
        empPhoto = helper.getImage(imgURL, imageFilePath)
        head, brand, color, genderid, meta, sku, isActive, price = helper.parseAsosFields(soup, url)

        numImagesDown += 1
        logger.info('Image number %s: %s' % (trendOrder, imageFilePath.split(os.sep)[-1]))

        # Create new entry in PRODUCT table
        helper.addNewProduct(site, searchTerm, imageFilePath, empPhoto, url, imgURL, head, color, genderid, brand, meta, sku, isActive, price)

        # Create new entry in ProductHistory table
        helper.addNewProductHistory(url, referenceOrder, trendOrder, price, searchTerm)

    else:
        # Get trending products
        trendDF = helper.resultDataframeAsos(urlReceived, 'trend', breakPointNumber=breakPointNumber)
        # Get all relevant results for searh term
        refDF = helper.resultDataframeAsos(urlReceived, 'reference', filterDF=trendDF) 

        # Iterate trending products
        for _, row in trendDF.iterrows():
            trendOrder = row['trendOrder']
            url = row['URL']
            imgURL = row['imgURL']
            price = row['price']
            # Retrieve reference order
            if not refDF.loc[refDF['URL']==url, 'referenceOrder'].empty:
                referenceOrder = refDF.loc[refDF['URL']==url, 'referenceOrder'].values[0]
            else:
                referenceOrder = 0
            # Check if url already exists in the PRODUCT table
            # If TRUE update the latest record in ProductHistory table
            # Otherwise download product image and create new product entry in Product and ProductHistory tables
            url = url.replace("'", "''")
            querydf = pd.read_sql_query("SELECT * FROM %s.dbo.Product WHERE  CONVERT(VARCHAR(MAX), %s.dbo.PRODUCT.URL) = \
                CONVERT(VARCHAR(MAX),'%s')" % (dbName, dbName, str(url)), engine)
            # querydf = pd.read_sql_query("SELECT * FROM public.\"Product\" WHERE public.\"Product\".\"URL\" = '%s'" %  url, engine)
            if not querydf.empty:
                # Update ProductHistory
                prdno = querydf['Oid'].values[0]
                helper.updateProductHistory(prdno, referenceOrder, trendOrder, price, url, searchTerm)
                numUrlExist += 1
                logger.info('Info for product %s updated' % prdno)
            else:
                # Download product image
                imageFilePath = helper.setImageFilePath(standardUrl, ''.join(searchTerm.split()), trendOrder)
                empPhoto = helper.getImage(imgURL, imageFilePath)
                numImagesDown += 1
                logger.info('Image number %s: %s' % (trendOrder, imageFilePath.split(os.sep)[-1]))

                # Find fields from product's webpage
                soup = helper.get_content(url)
                head, brand, color, genderid, meta, sku, isActive, price = helper.parseAsosFields(soup, url)
                # Create new entry in PRODUCT table
                helper.addNewProduct(site, folderName, imageFilePath, empPhoto, url, imgURL, head, color, genderid, brand, meta, sku, isActive, price)

                # Create new entry in ProductHistory table
                helper.addNewProductHistory(url, referenceOrder, trendOrder, price, searchTerm)

    logger.info('Images requested: %s,   Images Downloaded: %s (%s%%),   Images Existed: %s' % (
        breakPointNumber, numImagesDown, round(numImagesDown/breakPointNumber,2 ) * 100, numUrlExist))
    # The time needed to scrape this query


############ Main function ############
if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    logger = S4F_Logger('AsosLogger', user=user).logger
    helper = Helper(logger)

    start_time_all = time.time()

    currendDir = config.WEB_CRAWLERS
    engine = config.ENGINE
    dbName = config.DB_NAME
    # Webpage URL
    standardUrl = 'https://www.asos.com/uk/search/?q='
    site = str((standardUrl.split('.')[1]).capitalize())

    query = standardUrl + '%20'.join(searchTerm.split())
    logger.info('Parsing: ' + str(query))

    folderName = helper.getFolderName(searchTerm)
    performScraping(query, searchTerm, breakPointNumber=threshold)
    logger.info("Time to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
