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
    crawlSearchID, searchTerm, threshold, user = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
    start_time_all = time.time()

    currendDir = config.WEB_CRAWLERS

    # ==========================================================================================
    # Login and Scrape Instagram
    # ==========================================================================================
    username = config.INSTAGRAM_USERNAME
    password = config.INSTAGRAM_PASSWORD
    
    instagram = InstagramCrawler(username, password, user)
    logger = instagram.logger
    helper = instagram.helper
    instagram.login()

    insta = instagram.search_query(searchTerm=searchTerm, threshold=threshold)
    if len(insta) > 0:
        # Store results in the database ranked by the relevance of the experts terminology
        dataDF = save_ranked(helper, insta, adapter='Instagram')

        logger.info('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, 
                len(dataDF), round(len(dataDF)/threshold, 2 ) * 100)) 
        logger.info("Time to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
    else:
        logger.warning('No results in Instagram for %s' % searchTerm)