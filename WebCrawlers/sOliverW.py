# -*- coding: utf-8 -*-
########################################### Import all the libraries needed ###########################################
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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains


########################################### This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ###########################################
def performScraping(urlReceived, category):
    # Start time for counting how long does it take a query to scrape
    start_time = time.time()
    engine = sqlalchemy.create_engine(config['db_connection'] + config['db_name'])
    numUrlExist = 0
    numUrlSkipped = 0
    numImagesDown = 0

    ########################################### Loop Parameters ###########################################
    breaki = 0  # Counter for images per query

    # This will hold the url address
    url = urlReceived
    soup = helper_functions.makesOliverSoup(url)
    page = soup.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
    print('Scraping category %s at: %s' % (category, urlReceived))

    # Count of products proceeded for specific query
    i = 0

    # Flag if maxitem = 1 was a returned result
    ch = False

    # Check the maximum number of returned results (only number from text)
    try:
        maxitems = int(soup.find('span', {'class': re.compile("jsFiltersResultCount")}).text.replace('.', ''))
    except:
        maxitems = 1
        ch = True

    breakPointNumber = maxitems

    # Length of products in a single page (constant)
    plen = len(page)

    # Sum of our length of products in all pages (Initialized at length of first page)
    suml = len(page)

    trendDF = helper_functions.trendDataframesOliver(urlReceived, breakPointNumber)
    # This statement checks the number of images that were restricted to which were supposed to be downloaded
    while breaki < breakPointNumber:
        ########################################### Get next product page ###########################################
        if breakPointNumber >= suml and breaki >= suml:
            # Check if the result is only one
            if breakPointNumber > 1:                
                checkneNext = soup.find('a', {'class': re.compile('pagination__next jsClickTrack')})
                if checkneNext:
                    suml = suml + plen
                    temp = checkneNext.get('href').rsplit('=', 1)
                    new = temp[0] + '=' + str(suml)
                    soup = helper_functions.makesOliverSoup(new)
                    page = soup.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
                    i = 0

        # Count for stopping the while loop when we reach the desired number of results
        breaki = breaki + 1
        
        ########################################### Get the url of the ith element of search page ###########################################
        try:
            np = page[i].find('a', {'class': re.compile("js-ovlistview-productdetaillink")}).get('href')  # Find parent node of img
            if ('%.12s' % np) != 'https://www.':
                np = helper_functions.hyphen_split(urlReceived) + '/en' + np
            urlshort = np.rsplit('?', 1)[0]
        except:
            if maxitems == 1 and ch:
                np = url  # Find parent node of img
                urlshort = np.rsplit('?', 1)[0]
            numUrlSkipped += 1
            print('Something wrong with np')

        ########################################### Check if the url already exists and if something (Price, Ranking) has changed ###########################################
        ########################################### Read old data from database ###########################################
        ASK_SQL_Query = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(urlshort), engine)

        ########################################### Check if the url already exists and if something (Price, Ranking) has changed ###########################################
        # Dataframe with the results for the specific url from database
        testdf = pd.DataFrame(ASK_SQL_Query)

        # Find if url exists in trending df and in which row
        resultDF = trendDF.loc[trendDF['Url'] == urlshort]
        if resultDF.empty:
            trendrorder = None
        else:
            trendrorder = resultDF['TrendOrder'].values[0]

        # Check if url already exists
        exists = False
        if not testdf.empty:
            exists = True
            prdno = testdf['ProductNo'].values[0]
            flagprice = False

            ########################################### The new prices ###########################################
            try:
                newprice = page[i].find('span', {'class': re.compile('ta_price')}).text.replace(',', '.')  # Getting the price of the product
                newprice = float(re.findall("[+-]?\d+\.\d+", newprice)[0])
            except:
                flagprice = True
                newprice = None

            # Because different names for 2 types
            if flagprice:
                try:
                    newprice = page[i].find('span', {'class': re.compile('plproduct__saleprice ta_newPrice')}).text.replace(',', '.')  # Getting the price of the product
                    newprice = float(re.findall("[+-]?\d+\.\d+", newprice)[0])
                except:
                    newprice = None

            # Keep the most recent entry of the url in database PRODUCTHISTORY
            ASK_SQL_Query = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCTHISTORY WHERE SocialMedia.dbo.PRODUCTHISTORY.ProductNo = '{}'".format(prdno), engine)
            testdf2 = pd.DataFrame(ASK_SQL_Query)
            recent_date = testdf2['SearchDate'].max()
            # One row dataframe with the latest input fir the specific ProductNo
            recent = testdf2.loc[testdf2['SearchDate'] == recent_date]

            # Handle none values on trending order
            try:
                to = int(recent['TrendingOrder'])
            except:
                to = None
            # Check if the current value changed
            if float(recent['Price']) != newprice or (int(recent['ReferenceOrder']) != breaki) or (to != trendrorder):
                # The number of images - url  existed in the end of query
                numUrlExist = numUrlExist + 1
                dftemp2 = pd.DataFrame([{'ProductNo': prdno, 'ReferenceOrder': breaki, 'TrendingOrder': trendrorder, 'Price': newprice}])
                dftemp2.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)

        # If url doesn't exist begin to scrape
        if not exists:
            print('New entry for %s' % np)
            try:
                soup2 = helper_functions.makesOliverSoup(np)
            except:
                numUrlSkipped += 1
                print('Something wrong with np 2')

            ########################################### url ###########################################
            now = datetime.now()
            dt_string = now.strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
            image = soup2.find('img', {'srcset': re.compile('front')})
            nameOfFile = 'image' + str(breaki) + "_" + re.sub("[ /:]", ".", dt_string)
            print("Image number ", breaki, " : ", nameOfFile)

            try:
                img = image.get('srcset')
            except:
                print('Something is going wrong with src')
                i = i + 1
                # The number of images skipped in the end of query
                numUrlSkipped += 1
                continue

            if ('%.6s' % img) != 'https:':
                img = "https:" + img
            ImageType = ".jpeg"
            folderIndividualName = os.path.join(cwd, 'temp')  # Creates the path where the images will be stored

            ########################################## Create or Access a Folder to download an image ###########################################
            # Create The folder according to search name
            if not os.path.exists(folderIndividualName):
                os.makedirs(folderIndividualName)
            r = requests.get(img, allow_redirects=True)
            imageFile = open(os.path.join(folderIndividualName, nameOfFile + ImageType), 'wb')
            imageFile.write(r.content)
            imageFile.close()

            # The number of images downloaded in the end of query
            numImagesDown = numImagesDown + 1

            ########################################### Attributes ###########################################
            ########################################### cur_price, init_price - Clothe's current value and initial value (different only when product on sale) ###########################################
            flagprice = False
            try:
                price = soup2.find('span', {'class': re.compile('ta_price')}).text.replace(',', '.')  # Getting the price of the product
                price = float(re.findall("[+-]?\d+\.\d+", price)[0])
            except:
                flagprice = True
                price = None
            # Because different names for 2 types
            if flagprice:
                try:
                    price = soup2.find('span',{'class': re.compile('ta_newPrice')}).text.replace(',', '.')  # Getting the price of the product
                    price = float(re.findall("[+-]?\d+\.\d+", price)[0])
                except:
                    price = None
            ########################################### head - Clothe's General Description ###########################################
            try:
                head = soup2.find('h1', {'class': re.compile('o-copy o-copy--m m-buyBox__title ta_shortDescription')}).text
                head = re.sub('\s+', ' ', head).strip()
            except:
                head = ''
                print("Head is Empty")
            ########################################### brand - Clothe's Brand ###########################################
            brand = 'S.Oliver'
            ########################################### color - Clothe's Color ###########################################
            try:
                color = soup2.find('span', {'class': re.compile("o-copy o-copy--xs o-copy--bold m-buyBox__optionsSelected ta_colorName")}).text
                color = color.strip()
            except:
                color = ''
                print("Color is Empty")
            ########################################### gender - Clothe's Gender ###########################################
            try:
                gender = soup2.find('a', {'class': re.compile('o-link o-link--xxs c-breadcrumbs__link jsClickTrack')}).text
                gender = gender.strip()
                if gender == 'Men':
                    genderid = 1
                elif gender == 'Women':
                    genderid = 2
                elif gender == 'Junior':
                    genderid = 3
            except:
                gender = ''
                genderid = None
                print("Gender is Empty")
            ########################################### Other attributes ###########################################
            at = ''

            # For DETAILS
            try:
                t1 = soup2.findAll('div', {'class': re.compile('c-productInformation__listContainer')})[0]
                # Text of description title
                at = 'PRODUCT DETAILS:'
                # Find all lists from first tab (Product details)
                t3 = t1.findAll('ul')
                for u in t3:
                    # Take all elements for every list
                    for l in u.findAll('li'):
                        tem = l.text.strip()
                        tem = tem.replace('\n', ' ')
                        at = at + ' - ' + tem
            except:
                print('Details is Empty')

            # For FIT
            try:
                # Find all from second tab (FIT)
                t1 = soup2.findAll('div', {'class': re.compile('c-productInformation__listContainer')})[1]
                # Text of description title
                at = at + ' - FIT:'
                # Find all lists from second tab (FIT)
                t3 = t1.findAll('ul')
                for u in t3:
                    # Take all elements for every list
                    for l in u.findAll('li'):
                        tem = l.text.strip()
                        tem = tem.replace('\n', ' ')
                        at = at + ' - ' + tem
            except:
                print('Fit is Empty')

            # For MATERIAL & CARE INSTRUCTIONS
            try:
                # Text of description title
                at = at + ' - MATERIAL & CARE INSTRUCTIONS:'
                # for internal tabs of Material & Care Instructions
                for i in range(2, 5):
                    # Find all from second tab (FIT)
                    t1 = soup2.findAll('div', {'class': re.compile('c-productInformation__listContainer')})[i]
                    # Find all lists from third tab (MATERIAL & CARE INSTRUCTIONS)
                    t3 = t1.findAll('ul')
                    for u in t3:
                        # Take all elements for every list
                        for l in u.findAll('li'):
                            tem = l.text.strip()
                            tem = tem.replace('\n', ' ')
                            at = at + ' - ' + tem
            except:
                print('Details & Care Instructions is Empty')

            at = re.sub('\s+', ' ', at).strip()  # get rid of tabs, extra spaces
            site = str((urlReceived.split('.')[1]).capitalize())
            imagepath = os.path.join(cwd, 'temp', nameOfFile + ImageType)
            empPhoto = helper_functions.convertToBinaryData(imagepath)
            os.remove(imagepath)
            dftemp = pd.DataFrame([{'Crawler': site, 'SearchWords': category, 'Image': imagepath, 'ImageBlob': empPhoto, 'url': np.rsplit('?', 1)[0], 'ImageSource': img,
                                    'SiteClothesHeadline': head, 'Color': color, 'GenderID': genderid, 'Brand': brand, 'Metadata': at, 'ProductCategoryID': None, 'ProductSubcategoryID': None,
                                    'LengthID': None, 'SleeveID': None, 'CollarDesignID': None, 'NeckDesignID': None, 'FitID': None, 'ClusterID': -1, 'FClusterID': -1}])
            dftemp.to_sql("PRODUCT", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)
            ASK_SQL_Query = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(urlshort), engine)
            testdf3 = pd.DataFrame(ASK_SQL_Query)
            prno = testdf3['ProductNo'].values[0]
            dftemp2 = pd.DataFrame([{'ProductNo': prno, 'ReferenceOrder': breaki, 'TrendingOrder': trendrorder, 'Price': price}])
            dftemp2.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)
        i = i + 1
    os.rmdir(folderIndividualName)
    print('Images Wanted: %s, Images Downloaded: %s, Images Skipped: %s (%s), Images Existed: %s' %  (str(breakPointNumber), str(numImagesDown), str(round(numImagesDown/breakPointNumber * 100), 2), str(numUrlSkipped), str(numUrlExist)))
    # The time needed to scrape this query
    print("Time to scrape this query is %s seconds ---" % (time.time() - start_time), '\n')



########################################## The main function of our program ###########################################
if __name__ == '__main__':
    start_time_all = time.time()
    
    # Open project configuration file
    with open(helper_functions.PROJECT_CONFIG) as f:
        config = json.load(f)

    cwd = helper_functions.CWD
    breakNumber = -1
    categoryDF = pd.DataFrame(columns=['Category', 'CategoryUrl', 'Gender', 'SubGender'])
    gender = ['men/clothing', 'women/apparel', 'junior/boys/kids', 'junior/boys/teens', 'junior/girls/kids', 'junior/girls/teens', 'junior/babys/girls/', 'junior/babys/boys/']
    # Webpage URL
    for gend in gender:
        if 'women' in gend:
            gendid = 2
        elif 'men' in gend:
            gendid = 1
        elif 'boys' in gend:
            gendid = 3
        elif 'girls' in gend:
            gendid = 4
        standardUrl1 = 'https://www.soliver.eu/c/' + gend + '/'
        soup = helper_functions.clothCategorysOliver(standardUrl1)
        categ = soup.findAll('li', {'class': re.compile('secondary-nav__item')})
        for cat in categ:
            caturl = cat.find('a').get('href')
            caturl = 'https://www.soliver.eu' + caturl
            cattext = cat.text.strip()
            if gendid == 3 or gendid == 4:
                subgend = gend.split('/')[2]
                if 'boys' in subgend or 'girls' in subgend:
                    subgend = gend.split('/')[1]
            else:
                subgend = None
            print('Category: %s, URL: %s' % (cattext, caturl))
            series = pd.Series({'Category': cattext, 'CategoryUrl': caturl, 'Gender': gendid, 'SubGender': subgend}, index=categoryDF.columns)
            categoryDF = categoryDF.append(series, ignore_index=True)
    for row_index, caturl in categoryDF.iterrows():
        performScraping(caturl['CategoryUrl'], caturl['Category'])
    print("Time to scrape ALL queries is %s seconds ---" % (time.time() - start_time_all))

