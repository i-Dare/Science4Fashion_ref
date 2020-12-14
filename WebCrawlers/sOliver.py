# -*- coding: utf-8 -*-
############ Import all the libraries needed ############
import os
import json
import time
import requests
import sqlalchemy
import helper_functions
import pandas as pd
import regex as re

from bs4 import BeautifulSoup, ResultSet
from datetime import datetime


############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ############
def performScraping(urlReceived, keywords, breakPointNumber):
    # initialize scraping
    start_time = time.time()
    # stats counters
    numUrlExist = 0
    numImagesDown = 0
    ## Check no results page
    # Get search results page
    soup = helper_functions.get_content(urlReceived)
    noResultsMessage = soup.find('h1', {'class': re.compile(r'ov__titlename')})
    if noResultsMessage:
        print('Unfortunately, your search for produced no results.')
        return 0
    ## Get reference and trend order. Handle the case where the user enters the exact product 
    # name as search terms, and the webpage skips search results page and redirects to the product page
    try:
        # Get all relevant results for searh term
        refDF = helper_functions.resultDataframeSOliver(urlReceived, 'reference')
        # In case no 'breakpoint' fetch all the results
        if breakPointNumber==0:
            breakPointNumber = len(refDF) 
        # Get trending products
        trendDF = helper_functions.resultDataframeSOliver(urlReceived, 'trend', breakPointNumber)
        # Iterate trending products
    except Exception as e:        
        print('Exception: %s' % e)
        trendDF = pd.DataFrame(columns=['trendOrder', 'URL', 'imgURL'])
        refDF = pd.DataFrame(columns=['referenceOrder', 'URL', 'imgURL'])
        imgURL = soup.find('meta', {'property': re.compile('og:image')}).get('content').split('?')[0]
        series = pd.Series({'trendOrder':1, 'URL': urlReceived, 'imgURL': imgURL}, index=trendDF.columns)
        trendDF = trendDF.append(series, ignore_index=True)
        series = pd.Series({'referenceOrder':1, 'URL': urlReceived, 'imgURL': imgURL}, index=refDF.columns)
        refDF = refDF.append(series, ignore_index=True)
    
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
        soup = helper_functions.get_content(url)
        price, head, brand, color, genderid, meta, sku, isActive = helper_functions.parseSOliverFields(soup, url, imgURL)
        
        # Check if url already exists in the PRODUCT table
        # If TRUE update the latest record in ProductHistory table
        # Otherwise download product image and create new product entry in PRODUCT and ProductHistory tables
        url=url.replace("'", "''")
        # querydf = pd.read_sql_query("SELECT * FROM %s.dbo.PRODUCT WHERE %s.dbo.PRODUCT.url = '%s'" % (dbName, dbName, url), engine)
        querydf = pd.read_sql_query("SELECT * FROM public.\"Product\" WHERE public.\"Product\".\"URL\" = '%s'" %  url, engine)
        if not querydf.empty:
            # Update ProductHistory
            prdno = querydf['Oid'].values[0]
            helper_functions.updateProductHistory(prdno, referenceOrder, trendOrder, price, url)
            numUrlExist += 1
            print('Info for product %s updated' % prdno)
        else:
            # Download product image
            imageFilePath = helper_functions.setImageFilePath(standardUrl, keywords, trendOrder)
            empPhoto = helper_functions.getImage(imgURL, imageFilePath)
            numImagesDown += 1
            print('Image number %s: %s' % (trendOrder, imageFilePath.split(os.sep)[-1]))

            # Create new entry in PRODUCT table
            helper_functions.addNewProduct(site, folderName, imageFilePath, empPhoto, url, imgURL, head, color, genderid, brand, meta, sku, isActive, price)

            # Create new entry in ProductHistory table
            helper_functions.addNewProductHistory(url, referenceOrder, trendOrder, price)

    print('Images requested: %s,   Images Downloaded: %s (%s%%),   Images Existed: %s' % (
        breakPointNumber, numImagesDown, round(numImagesDown/breakPointNumber,2 ) * 100, numUrlExist))
    # The time needed to scrape this query
    print("\nTime to scrape this query is %s seconds ---" % round(time.time() - start_time, 2))


############ Main function ############
if __name__ == '__main__':
    start_time_all = time.time()
    
    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME
    # Webpage URL
    standardUrl = 'https://www.soliver.eu/search/?q='
    site = str((standardUrl.split('.')[1]).capitalize())

    ############ Open the file with read only permit ############
    file = open(os.path.join(currendDir, 'keywords.txt'), "r")

    ############ Use readlines to read all lines in the file ############
    lines = file.readlines()  # The variable "lines" is a list containing all lines in the file
    file.close()  # Close the file after reading the lines.

    ############ The File stores Input data as "<Number Of Images Required><<SPACE>><Search Text With Spaces>" ############
    for i in range(len(lines)):
        keys = lines[i]
        keys = keys.replace('\n', '')
        print('Crawler Search no. %s ------------------- Search query: %s' % (i + 1, keys))
        folderName = helper_functions.getFolderName(keys)

        keywords = keys.split(" ")
        try:
            breakNumber = int(keywords[0])
            keyUrl = standardUrl + '%20'.join(keywords[1:])
        except:
            breakNumber = 0
            keyUrl = standardUrl + '%20'.join(keywords)

        print('Page to be crawled: ' + str(keyUrl))
        print("Number of crawled images wanted: " + str(breakNumber))

        performScraping(keyUrl, folderName, breakNumber)
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
