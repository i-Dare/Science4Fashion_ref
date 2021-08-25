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

class SOliverCrawler:

    def __init__(self, crawlSearchID, searchTerm, threshold=10, user=config.DEFAULT_USER, 
            loglevel=config.DEFAULT_LOGGING_LEVEL):
        # Get input arguments
        self.crawlSearchID = crawlSearchID
        self.searchTerm = searchTerm
        self.threshold = int(threshold)
        self.user = user
        self.logging = S4F_Logger('SoliverLogger', user=user, level=loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)

        # Webpage URL
        self.standardUrl = 'https://www.soliver.eu/search/?q='
        self.site = str((self.standardUrl.split('.')[1]).capitalize())

        self.query_url = self.standardUrl + '%20'.join(self.searchTerm.split())
        self.logger.info('Parsing: ' + str(self.query_url), {'CrawlSearch': self.crawlSearchID})


    ############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the searchTerm text file ############
    def executeCrawling(self,):
        # initialize scraping
        start_time = time.time()
        productIDs = []
        ## Check no results page
        # Get search results page
        url, soup = self.helper.get_content(self.query_url, retry=5)
        noResultsMessage = soup.find('h1', {'class': re.compile(r'ov__titlename')})
        if noResultsMessage:
            self.logger.warning('Your search produced no results.', {'CrawlSearch': self.crawlSearchID})
            return 0
        ## Get reference and trend order. Handle the case where the user enters the exact product 
        # name as search terms, and the webpage skips search results page and redirects to the product page
        # In case no 'breakpoint' fetch 10 results
        if self.threshold==0:
            self.threshold= 10
        # Get trending products
        trendDF = self.helper.crawlingSOliver(self.query_url, 'trend', threshold=self.threshold)
        # Get all relevant results for searh term
        refDF = self.helper.crawlingSOliver(self.query_url, 'reference', filterDF=trendDF)
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
            url, soup = self.helper.get_content(url)
            price, head, brand, color, genderID, meta, sku, prodCatID, prodSubCatID = \
                    self.helper.parseSOliverFields(soup, url, imgURL, self.crawlSearchID)
            
            # Check if url already exists in the PRODUCT table
            # If TRUE update the latest record in ProductHistory table
            # Otherwise download product image and create new product entry in PRODUCT and ProductHistory tables
            uniq_params = {'table': 'Product', 'URL': url}
            params = {'table': 'Product', 'Description': self.searchTerm, 'Active':  True, 'Gender': genderID,
                    'ColorsDescription': color, 'Ordering': 0, 'ProductCode': sku, 'ProductTitle': head, 
                    'SiteHeadline': head, 'Metadata': meta, 'RetailPrice': price, 'URL': url, 
                    'ImageSource': imgURL, 'Brand': brand, 'ProductCategory': prodCatID, 
                    'ProductSubcategory': prodSubCatID}
            productID = self.helper.registerData(self.crawlSearchID, self.site, self.standardUrl, referenceOrder, trendOrder,  
                    uniq_params, params)
            productIDs.append(productID)

        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                {'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to scrape this query is %s seconds ---" 
                % round(time.time() - start_time, 2), {'CrawlSearch': self.crawlSearchID})
        return productIDs