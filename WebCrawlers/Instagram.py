# Medium Blog useful
# https://medium.com/@adamaulia/crawling-instagram-using-instalooter-2791edb453ff
# Documentation
# https://instalooter.readthedocs.io/en/latest/instalooter/index.html

#from instalooter.looters import ProfileLooter
#looter = ProfileLooter('eilex_kyp')
import os
import time
import sys

from core.helper_functions import *
import core.config as config
from WebCrawlers.SocialMedia.SocialMediaCrawlers import InstagramCrawler, save_ranked

if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    start_time_all = time.time()

    currendDir = config.WEB_CRAWLERS
    engine = config.ENGINE
    dbName = config.DB_NAME

    # ==========================================================================================
    # Login and Scrape Instagram
    # ==========================================================================================
    username = config.INSTAGRAM_USERNAME
    password = config.INSTAGRAM_PASSWORD
    
    instagram = InstagramCrawler(username, password, user)
    logger = instagram.logger
    helper = instagram.helper
    instagram.login()

    insta = instagram.search_query(query=searchTerm, threshold=threshold)
    # Store results in the database ranked by the relevance of the experts terminology
    dataDF = save_ranked(helper, insta, adapter='Instagram')

    logger.info('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    logger.info("Time to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
