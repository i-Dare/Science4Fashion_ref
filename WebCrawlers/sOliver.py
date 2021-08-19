# -*- coding: utf-8 -*-
############ Import all the libraries needed ############
import os
import time
from core.helper_functions import *
import pandas as pd
import regex as re

from bs4 import BeautifulSoup, ResultSet
from datetime import datetime
import sys
from core.helper_functions import *
from core.logger import S4F_Logger
import core.config as config


############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the searchTerm text file ############
def performScraping(crawlSearchID, urlReceived, searchTerm, breakPointNumber):
    # initialize scraping
    start_time = time.time()
    # stats counters
    cnt = 0
    ## Check no results page
    # Get search results page
    url, soup = helper.get_content(urlReceived, retry=5)
    noResultsMessage = soup.find('h1', {'class': re.compile(r'ov__titlename')})
    if noResultsMessage:
        logger.warning('Your search produced no results.')
        return 0
    ## Get reference and trend order. Handle the case where the user enters the exact product 
    # name as search terms, and the webpage skips search results page and redirects to the product page
    # In case no 'breakpoint' fetch 10 results
    if breakPointNumber==0:
        breakPointNumber = 10
    # Get trending products
    trendDF = helper.crawlingSOliver(urlReceived, 'trend', breakPointNumber=breakPointNumber)
    # Get all relevant results for searh term
    refDF = helper.crawlingSOliver(urlReceived, 'reference', filterDF=trendDF)
    if len(trendDF)==0 or len(refDF)==0:
        logger.warning('No results found for terms %s in %s' % (searchTerm, urlReceived))
    else:
        # Iterate trending products
        for _, row in trendDF.iterrows():
            trendOrder, url, imgURL, price, color = row['trendOrder'], row['URL'], row['imgURL'], \
                    row['price'], row['color']
            # Retrieve reference order
            if not refDF.loc[refDF['URL']==url, 'referenceOrder'].empty:
                referenceOrder = refDF.loc[refDF['URL']==url, 'referenceOrder'].values[0]
            else:
                referenceOrder = 0
            # Find fields from product's webpage
            url, soup = helper.get_content(url)
            price, head, brand, color, genderID, meta, sku, prodCatID, prodSubCatID = \
                    helper.parseSOliverFields(soup, url, imgURL, crawlSearchID)
            
            # Check if url already exists in the PRODUCT table
            # If TRUE update the latest record in ProductHistory table
            # Otherwise download product image and create new product entry in PRODUCT and ProductHistory tables
            uniq_params = {'table': 'Product', 'URL': url}
            params = {'table': 'Product', 'Description': searchTerm, 'Active':  True, 'Gender': genderID,
                    'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                    'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                    'ImageSource': imgURL, 'Brand': brand, 'ProductCategory': prodCatID, 
                    'ProductSubcategory': prodSubCatID}
            cnt, productID = helper.registerData(crawlSearchID, site, standardUrl, referenceOrder, trendOrder, cnt, 
                    uniq_params, params)

        logger.info('Images requested: %s, Images needed: %s, Images Downloaded: %s (%s%%)' % \
            (breakPointNumber, len(trendDF), cnt, round(cnt/len(trendDF),2 ) * 100))
    # The time needed to scrape this query
    logger.info("Time to scrape this query is %s seconds ---" % round(time.time() - start_time, 2))



############ Main function ############
if __name__ == '__main__':
    # Get input arguments
    crawlSearchID, searchTerm, threshold, user, loglevel = (int(sys.argv[1]), sys.argv[2], 
            int(sys.argv[3]), sys.argv[4], sys.argv[5])
    logging = S4F_Logger('SoliverLogger', user=user, loglevel=loglevel)
    logger = logging.logger
    helper = Helper(logging)

    start_time_all = time.time()
    
    currendDir = config.WEB_CRAWLERS
 
    # Webpage URL
    standardUrl = 'https://www.soliver.eu/search/?q='
    site = str((standardUrl.split('.')[1]).capitalize())

    query_url = standardUrl + '%20'.join(searchTerm.split())
    logger.info('Parsing: ' + str(query_url))

    folderName = helper.getFolderName(searchTerm)
    performScraping(crawlSearchID, query_url, searchTerm, threshold)