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


########################################### This function will create a soup and returns which is the parsed html format for extracting html tags of the webpage ###########################################
def makeSoup(url):
    # This will load the webpage for the given url
    driver = webdriver.Chrome(executable_path=os.path.join(CWD, 'chromedriver.exe'))
    driver.get(url)
    time.sleep(2)
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    driver.close()
    return soup


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
