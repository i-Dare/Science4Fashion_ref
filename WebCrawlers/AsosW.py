# -*- coding: utf-8 -*-
##### Import all the libraries needed #####
import os
from bs4 import BeautifulSoup, ResultSet
import regex as re
import json
import time
import requests
from datetime import datetime
import pandas as pd
import sqlalchemy
import helper_functions


############ This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ############
def performScraping(urlReceived, category):

    # Start time for counting how long does it take a query to scrape
    start_time = time.time()
    # stats counters
    numUrlExist = 0
    numImagesDown = 0

    # Image download directory
    folderIndividualName = os.path.join(currendDir, 'temp')
    if not os.path.exists(folderIndividualName):
        os.makedirs(folderIndividualName)

    soup = helper_functions.get_content(urlReceived)
    # Get all relevant results for searh term
    refDF = helper_functions.resultDataframeAsos(urlReceived, 'reference')
    # Get trending products
    trendDF = helper_functions.resultDataframeAsos(urlReceived, 'trend')

    # Iterate trending products
    for i, row in trendDF.iterrows():
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
        url = url.replace("'", "''")        
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
            dt_string = datetime.now().strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
            nameOfFile = 'image' + str(i) + "_" + re.sub("[ /:]", ".", dt_string)
            ImageType = ".jpeg"
            imageFilePath = os.path.join(currendDir, 'temp', nameOfFile + ImageType)
            empPhoto = helper_functions.getImage(imgURL, imageFilePath)
            os.remove(imageFilePath)
            numImagesDown += 1
            print('Image number %s: %s' % (trendOrder, imageFilePath.split(os.sep)[-1]))

            ## Find fields from product's webpage
            soup = helper_functions.get_content(url)
            head, brand, color, genderid, meta, sku, isActive = helper_functions.parseAsosFields(soup, url)
            # Create new entry in PRODUCT table
            helper_functions.addNewProduct(
                site, category, imageFilePath, empPhoto, url, imgURL, head, color, genderid, brand, meta, sku, isActive)

            # Create new entry in ProductHistory table
            helper_functions.addNewProductHistory(url, referenceOrder, trendOrder, price)

    os.rmdir(folderIndividualName)
    print('Images requested: %s,   Images Downloaded: %s (%s%%),   Images Existed: %s' % (len(trendDF), numImagesDown, round(numImagesDown/len(trendDF),2 ) * 100, numUrlExist))
    # The time needed to scrape this query
    print("\nTime to scrape category %s: %s seconds ---" % (category, round(time.time() - start_time, 2)))



############ Main function ############
if __name__ == '__main__':
    start_time_all = time.time()
    
    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME
    # Webpage URL
    standardUrl = 'https://www.asos.com/'
    site = str((standardUrl.split('.')[1]).capitalize())
    soup = helper_functions.get_content(standardUrl)
    categoryDF = pd.DataFrame(columns=['Category', 'CategoryUrl'])
    
    # Capture genders form header
    genders = []
    for li in soup.find('div', {'data-testid': 'header'}).findAll('li'):
        if li.find('a', {'id': re.compile('.+floor')}):
            genders.append(li.find('a', {'id': re.compile('.+floor')}).text)

    # Capture categories and category URLs form header
    category, categoryURLs, gensearch = [], [], []
    for gender in genders:
        genderURL = 'https://www.asos.com/' + gender
        soup = helper_functions.get_content(genderURL)
        buttons = soup.findAll('button')
        for b in buttons:
            for span in b.findAll('span', text=re.compile('cloth.+', re.IGNORECASE)):
                gensearch.append(span.findParent('button').get('data-id'))
                break
    for gender,search in zip(genders,gensearch):            
        catpage = soup.find('ul', {'data-id':re.compile(search)})
        categories = catpage.findAll('a', {'data-testid': re.compile("text-link")})
        for cat in categories:
            category.append(cat.text)
            categoryURLs.append(cat.get('href'))
 
    for cat,url in zip(category, categoryURLs):
        print('Category: %s, URL: %s' % (cat, url))
        performScraping(url, cat)

    print("Time to scrape ALL queries is %s seconds ---" % (time.time() - start_time_all))

