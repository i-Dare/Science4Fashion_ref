# -*- coding: utf-8 -*-
########################################### Import all the libraries needed ###########################################
import os
from bs4 import BeautifulSoup, ResultSet
import regex as re
from selenium import webdriver
import time
import pandas as pd


########################################### Variables shared amongst all crawlers ###########################################
CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')


########################################### Asos specific functionality ###########################################
########################################### This function will create a soup and returns which is the parsed html format for extracting html tags of the webpage ###########################################
def makeAsosSoup(url):
    # This will load the webpage for the given url
    driver = webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    driver.get(url)
    time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup


########################################### This function will create a soup for getting the trends html tags of the webpage ###########################################
def makeAsosSoupTrend(url):
    # This will load the webpage for the given url
    driver =  webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
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
def trendDataframeAsos(urlReceived, breakPointNumber):
    trendDF = pd.DataFrame(columns=['TrendOrder', 'Url'])
    order = 0
    flag = True
    # This will hold the url address
    url = urlReceived
    try:
        soupt = makeAsosSoupTrend(url)
        page = soupt.findAll('a', {'class': re.compile("_3TqU78D")})
    except:
        flag = False
        print('No trending attributes')
    while order < breakPointNumber and flag:
        if order > 0:
            new = soupt.find('a', {'data-auto-id': re.compile('loadMoreProducts')}).get('href')
            soupt = makeAsosSoup(new)
            page = soupt.findAll('a', {'class': re.compile("_3TqU78D")})
        for product in page:
            order = order + 1
            series = pd.Series({'TrendOrder': order, 'Url': product.get('href').rsplit('?',1)[0]}, index=trendDF.columns)
            trendDF = trendDF.append(series, ignore_index=True)
            if order == breakPointNumber:
                break
    return trendDF

def clothCategoryAsos(url):
    # This will load the webpage for the given url
    driver = webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    print('Open category menu')
    driver.maximize_window()
    driver.get(url)
    time.sleep(2)
    # driver.find_element_by_xpath(".//button[contains(.,'Clothing')]").click()
    # time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup


########################################### sOliver specific functionality ###########################################
########################################### This function will create a soup and returns which is the parsed html format for extracting html tags of the webpage ###########################################
def makesOliverSoup(url):
    # This will load the webpage for the given url
    driver = webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    driver.get(url)
    time.sleep(2)
    try:
        driver.find_element_by_xpath(".//a[contains(@class, 'jsPrivacyBarSubmit')]").click()
        time.sleep(2)
    except:
        print('No privacy message')
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup


########################################### This function will create a soup for getting the trends html tags of the webpage ###########################################
def makesOliverSoupTrend(url, r):
    # This will load the webpage for the given url
    driver =  webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    driver.get(url)
    time.sleep(2)
    try:
        driver.find_element_by_xpath(".//a[contains(@class, 'jsPrivacyBarSubmit')]").click()
        time.sleep(2)
        if r == 0:
            driver.find_element_by_xpath(".//div[@class= 'filterdropdown__toggle-label filterdropdownsort__toggle-label']").click()
            time.sleep(2)
            driver.find_element_by_xpath(".//button[contains(.,'New arrivals')]").click()
            time.sleep(2)
        # Get scroll height.
        height = driver.execute_script("return document.body.scrollHeight")
        new_height = 0
        while True:
            new_height = new_height + 2560
            # Scroll down to the bottom.
            driver.execute_script("window.scrollTo(0," + str(new_height) +")")
            # Wait to load the page.
            time.sleep(2)
            if new_height > height:
                break
    except:
        print('No results')
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup


########################################### This function will create a dataframe with trend ranking###########################################
def trendDataframesOliver(urlReceived, breakPointNumber):
    trendDF = pd.DataFrame(columns=['TrendOrder', 'Url'])
    order = 0
    # This will hold the url address
    soupt = makesOliverSoupTrend(urlReceived, 0)
    # Search for pagination and navigate product pages using pagination
    baseURL = urlReceived
    if soupt.find('a', {'class': re.compile('pagination__next jsClickTrack')}):
        # Get base URL
        baseURL = soupt.find('a', {'class': re.compile('pagination__next jsClickTrack')}).get('href').rsplit('=', 1)[0]
    for page in soupt.findAll('a', {'class': re.compile("pagination__link")}):
        # Get "start" parameter of pagination
        pagination = int(page.get('data-pagingparams').split(':')[-1][:-1])
        # Create new page URL
        new = '%s=%s' % (baseURL, pagination)
        print('Parsing new page: %s' % new)
        soup = makesOliverSoupTrend(new, 1)
        products = soup.findAll('div', {'class': re.compile("productlist__product js-ovgrid-item")})
        # Iterate products in the page
        for product in products:
            order += 1 # increament TrendOrder
            productInfo = product.findAll('a', {'class': re.compile("js-ovlistview-productdetaillink")})
            if productInfo:                
                urlt = 'https://www.soliver.eu' + productInfo[0].get('href')
                series = pd.Series({'TrendOrder': order, 'Url': urlt.rsplit('?',1)[0]}, index=trendDF.columns)
                # Update trend DateFrame
                trendDF = trendDF.append(series, ignore_index=True)
    return trendDF

def clothCategorysOliver(url):
    # This will load the webpage for the given url
    driver = webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    print('Open category menu')
    driver.maximize_window()
    driver.get(url)
    time.sleep(2)
    try:
        driver.find_element_by_xpath(".//a[contains(@class, 'jsPrivacyBarSubmit')]").click()
        time.sleep(2)
    except:
        print('No privacy message')
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup

########################################### Common functionality ###########################################
########################################### This function returns the folder name removing the number of images range from they line of keywords file ###########################################
def getFolderName(wholeName):
    tempArray = wholeName.split(" ")
    nameTemp = ""
    for i in range(1, len(tempArray)):
        nameTemp = nameTemp + " " + tempArray[i].strip('"')
    return nameTemp


###########################################  This function returns the 3rd appearance of / ###########################################
def hyphen_split(a):
    if a.count("/") == 1:
        return a.split("/")[0]
    else:
        return "/".join(a.split("/", 3)[:3])


def convertToBinaryData(filename):
    #Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData
