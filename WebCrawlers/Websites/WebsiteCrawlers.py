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
        self.logger.info('Parsing: ' + str(self.query_url), extra={'CrawlSearch': self.crawlSearchID})


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
            self.logger.warning('Failed to parse %s' % self.query_url, extra={'CrawlSearch': self.crawlSearchID})
        
        # If timeout or no result is returned
        try:
            noResultsMessage = soup.find('link', {'href': re.compile(r'no-results-page')})
            if noResultsMessage:
                self.logger.warning('Your search produced no results.', extra={'CrawlSearch': self.crawlSearchID})
                return 0
        except:
            return None

        # If soup is not redirected to a search page
        #    parse single product
        # else 
        #   parse according to reference and trending order
        if 'Search' not in str(soup.title) :
            self.logger.warning('Redirected to product page', extra={'CrawlSearch': self.crawlSearchID})
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
            productID = self.helper.registerData(self.crawlSearchID, self.standardUrl, referenceOrder, trendOrder, 
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
                productID = self.helper.registerData(self.crawlSearchID, self.standardUrl, referenceOrder, 
                        trendOrder, uniq_params, params)
                productIDs.append(productID)

        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                 extra={'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to complete query: %s seconds ---" % 
                round(time.time() - start_time, 2), extra={'CrawlSearch': self.crawlSearchID})
        return productIDs


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
        self.logger.info('Parsing: ' + str(self.query_url), extra={'CrawlSearch': self.crawlSearchID})


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
            self.logger.warning('Your search produced no results.', extra={'CrawlSearch': self.crawlSearchID})
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
            productID = self.helper.registerData(self.crawlSearchID, self.standardUrl, referenceOrder, trendOrder,  
                    uniq_params, params)
            productIDs.append(productID)

        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                extra={'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to complete query: %s seconds ---" 
                % round(time.time() - start_time, 2), extra={'CrawlSearch': self.crawlSearchID})
        return productIDs


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
        self.logger.info('Parsing: ' + str(self.query_url), extra={'CrawlSearch': self.crawlSearchID})


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
            self.logger.warning('Your search produced no results.', extra={'CrawlSearch': self.crawlSearchID})
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
            productID = self.helper.registerData(self.crawlSearchID, self.standardUrl, referenceOrder, trendOrder,
                    uniq_params, params)
            productIDs.append(productID)
        self.logger.info('Images requested: %s, New images found: %s' % (self.threshold, len(trendDF)),
                 extra={'CrawlSearch': self.crawlSearchID})
        # The time needed to scrape this query
        self.logger.info("Time to complete query: %s seconds ---" 
                % round(time.time() - start_time, 2), extra={'CrawlSearch': self.crawlSearchID})
        return productIDs
