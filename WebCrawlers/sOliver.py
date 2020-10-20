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

# temporaty imports
# import psycopg2

############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ############
def performScraping(urlReceived, keywords, breakPointNumber):
    # initialize scraping
    start_time = time.time()
    # stats counters
    numUrlExist = 0
    numImagesDown = 0

    # Get all relevant results for searh term
    refDF = helper_functions.resultDataframeSOliver(urlReceived, 'reference')
    # Get trending products
    trendDF = helper_functions.resultDataframeSOliver(urlReceived, 'trend', breakPointNumber)
    # Iterate trending products
    for i, row in trendDF.iterrows():
        trendOrder = row['trendOrder']
        url = row['Url']
        imgURL = row['imgURL']
        # Retrieve reference order
        referenceOrder = refDF.loc[refDF['Url']==url, 'referenceOrder'].values[0]
        # Find fields from product's webpage
        soup = helper_functions.get_content(url)
        price, head, brand, color, genderid, meta = helper_functions.parseSOliverFields(soup, url, imgURL)
        
        # Check if url already exists in the PRODUCT table
        # If TRUE update the latest record in PRODUCTHISTORY table
        # Otherwise download product image and create new product entry in PRODUCT and PRODUCTHISTORY tables
        # querydf = pd.read_sql_query(
        #     "SELECT * FROM public.\"PRODUCT\" WHERE public.\"PRODUCT\".url = '{}'".format(url), engine)
        querydf = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(url), engine)
        if not querydf.empty:
            # Update PRODUCTHISTORY
            prdno = querydf['ProductNo'].values[0]
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
            helper_functions.addNewProduct(
                site, folderName, imageFilePath, empPhoto, url, imgURL, head, color, genderid, brand, meta)

            # Create new entry in PRODUCTHISTORY table
            helper_functions.addNewProductHistory(url, referenceOrder, trendOrder, price)

    print('Images requested: %s,   Images Downloaded: %s (%s%%),   Images Existed: %s' % (
        breakPointNumber, numImagesDown, numImagesDown/breakPointNumber * 100, numUrlExist))
    # The time needed to scrape this query
    print("\nTime to scrape this query is %s seconds ---" % round(time.time() - start_time, 2))


########### The main function of our program ############
if __name__ == '__main__':
    start_time_all = time.time()
    
    cwd = helper_functions.CWD
    engine = helper_functions.engine
    # Webpage URL
    standardUrl = 'https://www.soliver.eu/search/?q='
    site = str((standardUrl.split('.')[1]).capitalize())

    ############ Open the file with read only permit ############
    file = open('keywords.txt', "r")

    ############ Use readlines to read all lines in the file ############
    lines = file.readlines()  # The variable "lines" is a list containing all lines in the file
    file.close()  # Close the file after reading the lines.

    ############ The File stores Input data as "<Number Of Images Required><<SPACE>><Search Text With Spaces>" ############
    for i in range(len(lines)):
        keys = lines[i]
        keys = keys.replace('\n', '')
        print('Crawler Search no. %s ------------------- Search query: %s' % (i + 1, keys))
        folderName = helper_functions.getFolderName(keys).replace(" ", "")

        keywords = keys.split(" ")

        breakNumber = int(keywords[0])
        keyUrl = standardUrl + '%20'.join(keywords[1:])

        print('Page to be crawled: ' + str(keyUrl))
        print("Number of crawled images wanted: " + str(breakNumber))

        performScraping(keyUrl, folderName, breakNumber)
    print("\nTime to scrape ALL queries is %s seconds ---" % (time.time() - start_time_all))
