#Testing Instagram Crawler
#Medium Blog useful
#https://medium.com/@adamaulia/crawling-instagram-using-instalooter-2791edb453ff
#Documentation
#https://instalooter.readthedocs.io/en/latest/instalooter/index.html

#from instalooter.looters import ProfileLooter
#looter = ProfileLooter('eilex_kyp')
import os
import time
import config
import helper_functions
from WebCrawlers.SocialMedia.SocialMediaCrawlers import InstagramCrawler, save_ranked


if __name__ == '__main__':
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
        query = keywords[1].strip('"')
        threshold = int(keywords[0])
        for j in range(2, keyLen):
            query = query + ' ' + keywords[j].strip('"')

        print('Query: ' + str(query))
        
        insta = instagram.search_query(query=query, threshold=threshold)
        # Store results in the database ranked by the relevance of the experts terminology
        dataDF = save_ranked(insta, instagram.home_page)

        print('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))





