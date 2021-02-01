import os
import time
import helper_functions
import config
from WebCrawlers.SocialMedia.SocialMediaCrawlers import PinterestCrawler, save_ranked
import sys


if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold = sys.argv[1], int(sys.argv[2])
    max_threshold = 250
    threshold = threshold if threshold <  max_threshold else 250
    
    start_time_all = time.time()
    
    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME    

    # ==========================================================================================
    # Login and Scrape Pinterest
    # ==========================================================================================
    username = config.PINTEREST_USERNAME
    password = config.PINTEREST_PASSWORD

    pinterest = PinterestCrawler(username=username, password=password)
    pinterest.login()

    pins = pinterest.search_query(query=searchTerm, threshold=threshold)
    # Store results in the database ranked by the relevance of the experts terminology
    dataDF = save_ranked(pins, adapter='Pinterest')

    print('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
