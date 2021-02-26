import os
import time
import sys

from  helper_functions import *
import config
from logger import S4F_Logger
from WebCrawlers.SocialMedia.SocialMediaCrawlers import PinterestCrawler, save_ranked
# from SocialMediaCrawlers import PinterestCrawler, save_ranked

if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    
    start_time_all = time.time()
    
    currendDir = config.WEB_CRAWLERS
    engine = config.ENGINE
    dbName = config.DB_NAME    

    # ==========================================================================================
    # Login and Scrape Pinterest
    # ==========================================================================================
    username = config.PINTEREST_USERNAME
    password = config.PINTEREST_PASSWORD

    pinterest = PinterestCrawler(username, password, user)
    logger = pinterest.logger
    helper = pinterest.helper
    pinterest.login()

    pins = pinterest.search_query(query=searchTerm, threshold=threshold)
    # Store results in the database ranked by the relevance of the experts terminology
    dataDF = save_ranked(helper, pins, adapter='Pinterest')

    logger.info('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    logger.info("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
