# -*- coding: utf-8 -*-
########### Import all the libraries needed ###########
import os
import time
import pandas as pd
import regex as re
import json

from bs4 import BeautifulSoup, ResultSet
from datetime import datetime
import sys
from core.helper_functions import *
from core.logger import S4F_Logger
import core.config as config


## This function handles the scraping functionality of the web crawler
def performScraping(urlReceived, searchTerm, breakPointNumber):
    # initialize scraping
    start_time = time.time()
    # stats counters
    cnt = 0
    ## Check no results page
    # Get search results page
    url, soup = helper.get_content(urlReceived)
    noResultsMessage = soup.find('span', {'class': re.compile(r'cat_subHeadline-11sbl')})
    if noResultsMessage:
        logger.info('Your search produced no results.')
        return 0
    ## Get reference and trend order. Handle the case where the user enters the exact product 
    # name as search terms, and the webpage skips search results page and redirects to the product page
     # In case no 'breakpoint' fetch 10 results
    if breakPointNumber==0:
        breakPointNumber = 10
    # Get trending products
    trendDF = helper.resultDataframeZalando(urlReceived, 'trend', breakPointNumber=breakPointNumber)
    # Get all relevant results for searh term
    refDF = helper.resultDataframeZalando(urlReceived, 'reference', filterDF=trendDF)
    # Iterate captured products
    for _, row in trendDF.iterrows():
        trendOrder = row['trendOrder']
        productPage, imgURL, brand, head, sku, price = row['URL'], row['imgURL'], row['Brand'], \
                row['Head'], row['SKU'], row['Price']
        # Retrieve reference order
        if not refDF.loc[refDF['URL']==productPage, 'referenceOrder'].empty:
            referenceOrder = refDF.loc[refDF['URL']==productPage, 'referenceOrder'].values[0]
        else:
            referenceOrder = 0
        
        paramsFromProductPage = helper.parseZalandoFields(productPage)
        paramsFromSearchPage = {'table': 'Product', 'Description': searchTerm, 'Active':  True, 
                'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 'SiteHeadline': head, 
                'RetailPrice': price, 'URL': productPage, 'ImageSource': imgURL, 'Brand': brand}
        uniq_params = {'table': 'Product', 'URL': productPage}
        params = dict(paramsFromProductPage, **paramsFromSearchPage)
        # Register product information
        cnt, productID = helper.registerData(site, standardUrl, referenceOrder, trendOrder, cnt,
                uniq_params, params)
    
    logger.info('Images requested: %s, Images needed: %s, Images Downloaded: %s (%s%%)' % \
            (breakPointNumber, len(trendDF), cnt, round(cnt/len(trendDF),2 ) * 100))
    # The time needed to scrape this query
    logger.info("Time to scrape this query is %s seconds ---" % round(time.time() - start_time, 2))


############ Main function ############
if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    logging = S4F_Logger('ZalandoLogger', user=user)
    logger = logging.logger
    helper = Helper(logging)

    start_time_all = time.time()
    
    currendDir = config.WEB_CRAWLERS
    engine = config.ENGINE
    dbName = config.DB_NAME
    # Webpage URL
    standardUrl = 'https://www.zalando.co.uk/api/catalog/articles?&query='
    site = str((standardUrl.split('.')[1]).capitalize())

    query = standardUrl + '%20'.join(searchTerm.split())
    logger.info('Parsing: ' + str(query))

    folderName = helper.getFolderName(searchTerm)
    performScraping(query, searchTerm, threshold)
    logger.info("Time to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
