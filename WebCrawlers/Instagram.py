# Medium Blog useful
# https://medium.com/@adamaulia/crawling-instagram-using-instalooter-2791edb453ff
# Documentation
# https://instalooter.readthedocs.io/en/latest/instalooter/index.html

#from instalooter.looters import ProfileLooter
#looter = ProfileLooter('eilex_kyp')
import os
import time
import sys

from  helper_functions import *
import config
from WebCrawlers.SocialMedia.SocialMediaCrawlers import InstagramCrawler, save_ranked


if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold, user, logfile = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    helper = Helper()
    logger = helper.initLogger('InstagramLogger', logfile)

    start_time_all = time.time()

    currendDir = helper.WEB_CRAWLERS
    engine = helper.ENGINE
    dbName = helper.DB_NAME

    # ==========================================================================================
    # Login and Scrape Instagram
    # ==========================================================================================
    username = config.INSTAGRAM_USERNAME
    password = config.INSTAGRAM_PASSWORD

    instagram = InstagramCrawler(username, password, user, logfile)
    instagram.login()

    insta = instagram.search_query(query=searchTerm, threshold=threshold)
    # Store results in the database ranked by the relevance of the experts terminology
    dataDF = save_ranked(helper, insta, adapter='Instagram')

    logger.info('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    logger.info("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
