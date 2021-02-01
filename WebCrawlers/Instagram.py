# Medium Blog useful
# https://medium.com/@adamaulia/crawling-instagram-using-instalooter-2791edb453ff
# Documentation
# https://instalooter.readthedocs.io/en/latest/instalooter/index.html

#from instalooter.looters import ProfileLooter
#looter = ProfileLooter('eilex_kyp')
import os
import time
import config
import helper_functions
from WebCrawlers.SocialMedia.SocialMediaCrawlers import InstagramCrawler, save_ranked
import sys


if __name__ == '__main__':
    # Get input arguments
    searchTerm, threshold = sys.argv[1], int(sys.argv[2])

    start_time_all = time.time()

    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME

    # ==========================================================================================
    # Login and Scrape Instagram
    # ==========================================================================================
    username = config.INSTAGRAM_USERNAME
    password = config.INSTAGRAM_PASSWORD

    instagram = InstagramCrawler(username=username, password=password)
    instagram.login()
        
    insta = instagram.search_query(query=searchTerm, threshold=threshold)
    # Store results in the database ranked by the relevance of the experts terminology
    dataDF = save_ranked(insta, adapter='Instagram')

    print('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
