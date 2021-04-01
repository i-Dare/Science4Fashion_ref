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


## This function handles the scraping functionality of the web crawler
def performScraping(urlReceived, searchTerm, breakPointNumber):
    # initialize scraping
    start_time = time.time()
    # stats counters
    cnt = 0
    ## Check no results page
    # Get search results page
    try:
        url, soup = helper.get_content(urlReceived, retry=5)
    except Exception as e:                
        logger.warn_and_trace(e)
        logger.warning('Failed to parse %s' % urlReceived)
    
    # If timeout or no result is returned
    try:
        noResultsMessage = soup.find('link', {'href': re.compile(r'no-results-page')})
        if noResultsMessage:
            logger.info('Your search produced no results.')
            return 0
    except:
        return None

    # If soup is not redirected to a search page
    #    parse single product
    # else 
    #   parse according to reference and trending order
    if 'Search' not in str(soup.title) :
        logger.warning('Redirected to product page')
        jsonUnparsed = soup.find('script', {'id': re.compile(r'split-structured-data')})
        product = json.loads(re.findall(r'({\".+})', str(jsonUnparsed))[0].replace('\\', ''))
        breakPointNumber = 1
        url = product['url']        
        imgURL = product['image']
        #  Download product image
        trendOrder = referenceOrder = 0

        # Register product information
        head, brand, color, genderid, meta, sku, price = helper.parseAsosFields(soup, url)
        uniq_params = {'table': 'Product', 'URL': url}
        params = {'table': 'Product', 'Description': searchTerm, 'Active':  True, 'Gender': genderid,
                'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                'ImageSource': imgURL, 'Brand': brand}
        cnt, productID = helper.registerData(site, standardUrl, referenceOrder, trendOrder, cnt, 
                uniq_params, params)
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
            # Retrieve reference order
            if not refDF.loc[refDF['URL']==url, 'referenceOrder'].empty:
                referenceOrder = refDF.loc[refDF['URL']==url, 'referenceOrder'].values[0]
            else:
                referenceOrder = 0
            # Register product information
            urlReceived, soup = helper.get_content(url, retry=3)
            head, brand, color, genderid, meta, sku, price = helper.parseAsosFields(soup, url)
            uniq_params = {'table': 'Product', 'URL': url}
            params = {'table': 'Product', 'Description': searchTerm, 'Active':  True, 'Gender': genderid,
                    'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                    'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                    'ImageSource': imgURL, 'Brand': brand}
            cnt, productID = helper.registerData(site, standardUrl, referenceOrder, 
                    trendOrder, cnt, uniq_params, params)

    logger.info('Images requested: %s, Images needed: %s, Images Downloaded: %s (%s%%)' % \
            (breakPointNumber, len(trendDF), cnt, round(cnt/len(trendDF),2 ) * 100))
    # The time needed to scrape this query
    logger.info("Time to scrape this query is %s seconds ---" % round(time.time() - start_time, 2))


############ Main function ############
if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    logging = S4F_Logger('AsosLogger', user=user)
    logger = logging.logger
    helper = Helper(logging)

    start_time_all = time.time()

    currendDir = config.WEB_CRAWLERS
    # Webpage URL
    standardUrl = 'https://www.asos.com/uk/search/?q='
    site = str((standardUrl.split('.')[1]).capitalize())

    query = standardUrl + '%20'.join(searchTerm.split())
    logger.info('Parsing: ' + str(query))

    folderName = helper.getFolderName(searchTerm)
    performScraping(query, searchTerm, breakPointNumber=threshold)
    logger.info("Time to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
