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

########################################### This function will create a soup for getting the trends html tags of the webpage ###########################################
def makeSoupTrend(url):
    # This will load the webpage for the given url
    driver =  webdriver.Chrome(executable_path=os.path.join(cwd, 'chromedriver.exe'))
    driver.get(url)
    time.sleep(2)
    driver.find_element_by_xpath(".//button[contains(.,'Sort')]").click()
    time.sleep(2)
    driver.find_element_by_xpath("//label[@for='plp_web_sort_whats_new']").click()
    time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup

########################################### This function will create a dataframe with trend ranking###########################################
def trendDataframe(urlReceived, breakPointNumber):
    trendDF = pd.DataFrame(columns=['TrendOrder', 'Url'])
    order = 0
    flag = True
    # This will hold the url address
    url = urlReceived
    try:
        soupt = makeSoupTrend(url)
        page = soupt.findAll('a', {'class': re.compile("_3TqU78D")})
    except:
        flag = False
        print('No trending attributes')
    while order < breakPointNumber and flag:
        if order > 0:
            new = soupt.find('a', {'data-auto-id': re.compile('loadMoreProducts')}).get('href')
            soupt = helper_functions.makeSoup(new)
            page = soupt.findAll('a', {'class': re.compile("_3TqU78D")})
        for product in page:
            order = order + 1
            series = pd.Series({'TrendOrder': order, 'Url': product.get('href').rsplit('?',1)[0]}, index=trendDF.columns)
            trendDF = trendDF.append(series, ignore_index=True)
            if order == breakPointNumber:
                break
    return trendDF

########################################### This function will be called every new keyword line is encountered and will start scraping the amazon web page of the search result according to the text mention in the keywords text file ###########################################
def performScraping(urlReceived, folderName, breakPointNumber):

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
    soup = helper_functions.makeSoup(url)
    page = soup.findAll('a', {'class': re.compile("_3TqU78D")})
    print("Folder Name for the images is: ", folderName)

    # Count of products proceeded for specific query
    i = 0

    # Flag for checking if the maxitem = 1 is not our responsibility
    ch = False

    # Check the maximum number of returned results (only number from text)
    try:
        maxitems = int(''.join(filter(str.isdigit, soup.find('p', {'data-auto-id': re.compile("styleCount")}).text)))
    except:
        maxitems = 1

    # If the query has zero results
    try:
        warn = soup.find('h2', {'class': re.compile('grid-text__title')}).text
        maxitems = 0
    except:
        warn = 'No warning'

    # Check if the number of items wanted exceed the returning results
    if breakPointNumber > maxitems:
        breakPointNumber = maxitems
        print('The number of results crawled from the site is smaller than the give one. New number of images: ' + str(breakPointNumber))

    # Length of products in a single page (constant)
    plen = len(page)

    # Sum of our length of products in all pages (Initialized at length of first page)
    suml = len(page)

    trendDF = trendDataframe(urlReceived, breakPointNumber)

    # This statement checks the number of images that were restricted to which were supposed to be downloaded
    while breaki < breakPointNumber:
        ########################################### Get next product page ###########################################
        if breakPointNumber >= suml and breaki >= suml:
            # Check if the result is only one
            if breakPointNumber > 1:
                new = soup.find('a', {'data-auto-id': re.compile('loadMoreProducts')}).get('href')
                soup = helper_functions.makeSoup(new)
                page = soup.findAll('a', {'class': re.compile("_3TqU78D")})
                suml = suml + plen
                i = 0

        # Count for stopping the while loop when we reach the desired number of results
        breaki = breaki + 1

        ########################################### Get the url of the ith element of search page ###########################################
        try:
            np = page[i].get('href')
            urlshort = np.rsplit('?', 1)[0]
            # Find parent node of img
            if ('%.12s' % np) != 'https://www.':
                np = hyphen_split(standardUrl) + '/en' + np
                urlshort = np.rsplit('?',1)[0]
        except:
            if maxitems == 1:
                np = url  # Find parent node of img
                urlshort = np.rsplit('?',1)[0]
            print('Something wrong with np')

        ########################################### Check if the url already exists and if something (Price, Ranking) has changed ###########################################
        ########################################### Read old data from database ###########################################
        ASK_SQL_Query = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(urlshort), engine)

        # Dataframe with the results for the specific url from database
        testdf = pd.DataFrame(ASK_SQL_Query)

        # Find if url exists in trending df and in which row
        resultDF = trendDF.loc[trendDF['Url'] == urlshort]
        print(resultDF)
        if resultDF.empty:
            trendrorder = None
        else:
            trendrorder = resultDF['TrendOrder'].values[0]

        # Check if url already exists
        exists = False
        if not testdf.empty:
            exists = True
            prdno = testdf['ProductNo'].values[0]
            flag1 = True
            ########################################### The new prices ###########################################
            try:
                newprice = float(re.findall("[+-]?\d+\.\d+", page[i].find('span', {'data-auto-id': re.compile("productTilePrice")}).text)[0])
                flag1 = False
                newprice = float(re.findall("[+-]?\d+\.\d+", page[i].find('span', {'data-auto-id': re.compile("productTileSaleAmount")}).text)[0])
            except:
                if flag1:
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
            print(float(recent['Price']))
            print(int(recent['ReferenceOrder']))
            print(to)
            if float(recent['Price']) != newprice or (int(recent['ReferenceOrder']) != breaki) or (to != trendrorder):
                # The number of images - url  existed in the end of query
                numUrlExist = numUrlExist + 1
                dftemp2 = pd.DataFrame([{'ProductNo': prdno, 'ReferenceOrder': breaki, 'TrendingOrder': trendrorder, 'Price': newprice}])
                dftemp2.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)

        # If url doesn't exist begin to scrape
        if not exists:
            try:
                soup2 = makeSoup(np)
            except:
                print('Something wrong with np 2')

            ########################################### url ###########################################
            now = datetime.now()
            dt_string = now.strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
            image = soup2.find('img', {'alt': re.compile('1 of 4')})
            nameOfFile = 'image' + str(breaki) + "_" + re.sub("[ /:]", ".", dt_string)
            print("Image number ", breaki, " : ", nameOfFile)

            try:
                img = image.get('src')
                # Small images, delete constraints from the tail of the src
                img = img.rsplit('?', 1)[0]
            except:
                print('Something is going wrong with src')
                i = i + 1
                # The number of images skipped in the end of query
                numUrlSkipped = numUrlSkipped + 1
                continue

            if ('%.6s' % img) != 'https:':
                img = "https:" + img
            ImageType = ".jpeg"
            imgSiteFolder = (standardUrl.split('.')[1]).capitalize() + 'Images'
            folderIndividualName = os.path.join(cwd, imgSiteFolder, folderName)  # Creates the path where the images will be stored

            ########################################## Create or Access a Folder to download an image ###########################################
            # Create The folder according to search name
            if not os.path.exists(folderIndividualName):
                os.makedirs(folderIndividualName)
            r = requests.get(img, allow_redirects=True)
            imageFile = open(folderIndividualName + nameOfFile + ImageType, 'wb')
            imageFile.write(r.content)
            imageFile.close()

            # The number of images downloaded in the end of query
            numImagesDown = numImagesDown + 1

            ########################################### Attributes ###########################################
            ########################################### cur_price, init_price - Clothe's current value and initial value (different only when product on sale) ###########################################
            try:
                price = soup2.find('span', {'data-id': re.compile('current-price')}).text  # Getting the price of the product
                price = float(re.findall("[+-]?\d+\.\d+", price)[0])
            except:
                price = None

            ########################################### head - Clothe's General Description ###########################################
            try:
                head = soup2.find('div', {'class': re.compile('product-hero')}).find('h1').text
                head = re.sub('\s+', ' ', head).strip()
            except:
                head = ''
                print("Head is Empty")
            ########################################### brand - Clothe's Brand ###########################################
            try:
                brand = soup2.find('div', {'class': re.compile('product-description')}).findAll('a')[1].text
            except:
                brand = ''
                print("Brand is Empty")
            ########################################### color - Clothe's Color ###########################################
            try:
                color = soup2.find('span', {'class': re.compile("product-colour")}).text
            except:
                color = ''
                print("Color is Empty")
            ########################################### gender - Clothe's Gender ###########################################
            try:
                gender = soup2.find('button', {'class': re.compile('TO7hyVB _3B0kHbC _3AH1eDT Tar7aO0')}).text
                if gender == 'MAN':
                    genderid = 1
                else:
                    genderid = 2
            except:
                gender = ''
                genderid = None
                print("Gender is Empty")

            ########################################### Other attributes ###########################################
            at = ''
            # For Description
            try:
                for li in soup2.find('div', {'class': re.compile('product-description')}).find('ul').findAll('li'):
                    temp = li.text.strip()
                    at = at + temp + " - "
            except:
                print('The Product Description is Empty')
            try:
                temp1 = soup2.find('div', {'class': re.compile('about-me')}).find('p').text
                if (temp1 != '' or temp1 != ' '):
                    at = at + temp1 + " - "
            except:
                print('No p type')
            # Second way of About Me
            try:
                for div in soup2.find('div', {'class': re.compile('about-me')}).findAll('div'):
                    temp = div.text.strip()
                    at = at + temp + " - "
            except:
                print('No div type')
            at = re.sub('\s+', ' ', at).strip()  # get rid of tabs, extra spaces
            site = str((standardUrl.split('.')[1]).capitalize())
            imagepath = os.path.join(cwd, imgSiteFolder, folderName, nameOfFile + ImageType)
            empPhoto = helper_functions.convertToBinaryData(imagepath)
            dftemp = pd.DataFrame([{'Crawler': site, 'SearchWords': folderName, 'Image': imagepath, 'ImageBlob': empPhoto, 'url': np.rsplit('?',1)[0], 'ImageSource': img,
                                    'SiteClothesHeadline': head, 'Color': color, 'GenderID': genderid, 'Brand': brand, 'Metadata': at, 'ProductCategoryID': None, 'ProductSubcategoryID': None,
                                    'LengthID': None, 'SleeveID': None, 'CollarDesignID': None, 'NeckDesignID': None, 'FitID': None, 'ClusterID': -1, 'FClusterID': -1}])
            dftemp.to_sql("PRODUCT", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)
            ASK_SQL_Query = pd.read_sql_query("SELECT * FROM SocialMedia.dbo.PRODUCT WHERE SocialMedia.dbo.PRODUCT.url = '{}'".format(urlshort), engine)
            testdf3 = pd.DataFrame(ASK_SQL_Query)
            prno = testdf3['ProductNo'].values[0]
            dftemp2 = pd.DataFrame([{'ProductNo': prno,'ReferenceOrder': breaki, 'TrendingOrder': trendrorder, 'Price': price}])
            dftemp2.to_sql("PRODUCTHISTORY", schema='SocialMedia.dbo', con=engine, if_exists='append', index=False)
        i = i + 1
    print('Images Wanted: ' + str(breakPointNumber) + ', Images Downloaded: ' + str(numImagesDown) + ' (' + str(numImagesDown/breakPointNumber * 100) + ') , Images Skipped: ' + str(numUrlSkipped) + ' , Images Existed: ' + str(numUrlExist))
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
    # Webpage URL
    standardUrl = 'https://www.asos.com/uk/search/?q='

    ########################################### Open the file with read only permit ###########################################
    file = open('keywords.txt', "r")

    ########################################### Use readlines to read all lines in the file ###########################################
    lines = file.readlines()  # The variable "lines" is a list containing all lines in the file
    file.close()  # Close the file after reading the lines.
    ########################################### The File stores Input data as "<Number Of Images Required><<SPACE>><Search Text With Spaces>" ###########################################
    for i in range(0, len(lines)):
        keys = lines[i]
        keys = keys.replace('\n', '')
        print("Crawler Search no." + str(i + 1) + ' ------------------- Search query: "' + str(keys) + '"') #
        folderName = helper_functions.getFolderName(keys).replace(" ", "")

        keywords = keys.split(" ")
        keyLen = len(keywords)

        breakNumber = int(keywords[0])
        keyUrl = standardUrl
        for j in range(1, keyLen):

            if keyUrl == standardUrl:
                keyUrl = keyUrl + keywords[j].strip('"')

            else:
                keyUrl = keyUrl + "+" + keywords[j].strip('"')
        print('Page to be crawled: ' + str(keyUrl))
        print("Number of crawled images wanted: " + str(breakNumber))

        performScraping(keyUrl, folderName, breakNumber)

    print("Time to scrape ALL queries is %s seconds ---" % (time.time() - start_time_all))

