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

class ZalandoCrawler:

    def __init__(self, crawlSearchID, searchTerm, threshold=10, user=config.DEFAULT_USER, 
            loglevel=config.DEFAULT_LOGGING_LEVEL):
        # Get input arguments
        self.crawlSearchID = crawlSearchID
        self.searchTerm = searchTerm
        self.threshold = int(threshold)
        self.user = user
        self.logging = S4F_Logger('ZalandoLogger', user=self.user, level=loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)

        # Webpage URL
        self.standardUrl = 'https://www.zalando.co.uk/api/catalog/articles?&query='
        self.site = str((self.standardUrl.split('.')[1]).capitalize())

        self.query_url = self.standardUrl + '%20'.join(searchTerm.split())
        self.logger.info('Parsing: ' + str(self.query_url), {'CrawlSearch': self.crawlSearchID})


    ## This function handles the scraping functionality of the web crawler
    def executeCrawling(self,):
        # initialize scraping
        start_time = time.time()
        productIDs = []
        ## Check no results page
        # Get search results page
        url, soup = self.helper.get_content(self.query_url)
        noResultsMessage = soup.find('span', {'class': re.compile(r'cat_subHeadline-11sbl')})
        if noResultsMessage:
            self.logger.warning('Your search produced no results.', {'CrawlSearch': self.crawlSearchID})
            return 0
        ## Get reference and trend order. Handle the case where the user enters the exact product 
        # name as search terms, and the webpage skips search results page and redirects to the product page
        # In case no 'breakpoint' fetch 10 results
        if self.threshold==0:
            self.threshold = 10
        # Get trending products
        trendDF = self.helper.crawlingZalando(self.query_url, 'trend', threshold=self.threshold)
        # Get all relevant results for searh term
        refDF = self.helper.crawlingZalando(self.query_url, 'reference', filterDF=trendDF)
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
            
            paramsFromProductPage = self.helper.parseZalandoFields(productPage, self.crawlSearchID)
            paramsFromSearchPage = {'table': 'Product', 'Description': self.searchTerm, 'Active':  True, 
                    'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 'SiteHeadline': head, 
                    'RetailPrice': price, 'URL': productPage, 'ImageSource': imgURL, 'Brand': brand}
            uniq_params = {'table': 'Product', 'URL': productPage}
            params = dict(paramsFromProductPage, **paramsFromSearchPage)
            # Register product information
            productID = self.helper.registerData(self.crawlSearchID, self.site, self.standardUrl, referenceOrder, trendOrder,
                    uniq_params, params)
            productIDs.append(productID)
        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                 {'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to scrape this query is %s seconds ---" 
                % round(time.time() - start_time, 2), {'CrawlSearch': self.crawlSearchID})
        return productIDs
