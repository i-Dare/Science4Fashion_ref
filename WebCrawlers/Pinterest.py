import os
import time
import helper_functions
import config
from WebCrawlers.PinterestCrawler.pinterest_with_requests import PinterestScraper


if __name__ == '__main__':
    start_time_all = time.time()
    # ==================================================================================================
    # Login and Scrape Pinterest
    # ==================================================================================================
    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME    
    
    username = config.PINTEREST_USERNAME
    password = config.PINTEREST_PASSWORD

    pinterest = PinterestScraper(username_or_email=username, password=password)
    logged_in = pinterest.login()

    if logged_in is True:
        print('Logged in successfully')
        
        
    ############ Open the file with read only permit ############
    file = open(os.path.join(currendDir, 'keywords.txt'), "r")

    ############ Use readlines to read all lines in the file ############
    lines = file.readlines()  # The variable "lines" is a list containing all lines in the file
    file.close()  # Close the file after reading the lines.
    
    for i in range(0, len(lines)):
        keys = lines[i]
        keys = keys.replace('\n', '')
        print('Crawler Search no. %s ------------------- Search query: %s' % (i + 1, keys))

        keywords = keys.split(" ")
        keyLen = len(keywords)
        keyUrl = keywords[1].strip('"')
        breakNumber = int(keywords[0])
        for j in range(2, keyLen):
            keyUrl = keyUrl + ' ' + keywords[j].strip('"')

        print('Query: ' + str(keyUrl))
        print("Number of crawled images wanted: " + str(breakNumber))
        pins = pinterest.search_pins(query=keyUrl, threshold=breakNumber)
        # Store results in the database ranked by the relevance of the experts terminology
        pinterest.rankedSave(pins)
                
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
