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
import core.config as config
from core.logger import S4F_Logger

class AsosCrawler:

    def __init__(self, crawlSearchID, searchTerm, threshold=10, user=config.DEFAULT_USER, 
            loglevel=config.DEFAULT_LOGGING_LEVEL):
        # Get input arguments
        self.crawlSearchID = crawlSearchID
        self.searchTerm = searchTerm
        self.threshold = int(threshold)
        self.user = user
        self.logging = S4F_Logger('AsosLogger', user=user, level=loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)

        # Webpage URL
        self.standardUrl = 'https://www.asos.com/uk/search/?q='
        self.site = str((self.standardUrl.split('.')[1]).capitalize())

        self.query_url = self.standardUrl + '%20'.join(self.searchTerm.split())
        self.logger.info('Parsing: ' + str(self.query_url), {'CrawlSearch': self.crawlSearchID})


    ## This function handles the scraping functionality of the web crawler
    def executeCrawling(self,):
        # initialize scraping
        start_time = time.time()
        productIDs = []
        ## Check no results page
        # Get search results page
        try:
            url, soup = self.helper.get_content(self.query_url, retry=5)
        except Exception as e:                
            self.logger.warn_and_trace(e)
            self.logger.warning('Failed to parse %s' % self.query_url, {'CrawlSearch': self.crawlSearchID})
        
        # If timeout or no result is returned
        try:
            noResultsMessage = soup.find('link', {'href': re.compile(r'no-results-page')})
            if noResultsMessage:
                self.logger.warning('Your search produced no results.', {'CrawlSearch': self.crawlSearchID})
                return 0
        except:
            return None

        # If soup is not redirected to a search page
        #    parse single product
        # else 
        #   parse according to reference and trending order
        if 'Search' not in str(soup.title) :
            self.logger.warning('Redirected to product page', {'CrawlSearch': self.crawlSearchID})
            self.threshold = 1
            #  Download product image
            trendOrder = referenceOrder = 0

            # Register product information
            head, brand, color, genderID, meta, sku, price, url, imgURL, prodCatID, prodSubCatID = self.helper.parseAsosFields(soup, url, self.crawlSearchID)
            uniq_params = {'table': 'Product', 'URL': url}
            params = {'table': 'Product', 'Description': self.searchTerm, 'Active':  True, 'Gender': genderID,
                    'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                    'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                    'ImageSource': imgURL, 'Brand': brand, 'ProductCategory': prodCatID, 'ProductSubCategory': prodSubCatID}
            productID = self.helper.registerData(self.crawlSearchID, self.site, self.standardUrl, referenceOrder, trendOrder, 
                    uniq_params, params)
            productIDs.append(productID)
        else:
            # Get trending products
            trendDF = self.helper.crawlingAsos(self.query_url, 'trend', threshold=self.threshold)
            # Get all relevant results for searh term
            refDF = self.helper.crawlingAsos(self.query_url, 'reference', filterDF=trendDF) 

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
                self.query_url, soup = self.helper.get_content(url, retry=3)
                head, brand, color, genderID, meta, sku, price, url, imgURL, prodCatID, prodSubCatID = self.helper.parseAsosFields(soup, url, self.crawlSearchID)
                uniq_params = {'table': 'Product', 'URL': url}
                params = {'table': 'Product', 'Description': self.searchTerm, 'Active':  True, 'Gender': genderID,
                        'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                        'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                        'ImageSource': imgURL, 'Brand': brand, 'ProductCategory': prodCatID, 'ProductSubCategory': prodSubCatID}
                productID = self.helper.registerData(self.crawlSearchID, self.site, self.standardUrl, referenceOrder, 
                        trendOrder, uniq_params, params)
                productIDs.append(productID)

        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                 {'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to scrape this query is %s seconds ---" % 
                round(time.time() - start_time, 2), {'CrawlSearch': self.crawlSearchID})
        return productIDs

    