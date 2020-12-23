import os
import time
import helper_functions
import config
from WebCrawlers.SocialMedia.SocialMediaCrawlers import PinterestCrawler, save_ranked


if __name__ == '__main__':
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

        pins = pinterest.search_query(query=query, threshold=threshold)
        # Store results in the database ranked by the relevance of the experts terminology
        dataDF = save_ranked(pins, adapter='Pinterest')

        print('Images requested: %s,   Images Downloaded: %s (%s%%)' % (threshold, len(dataDF), round(len(dataDF)/threshold,2 ) * 100)) 
    print("\nTime to scrape ALL queries is %s seconds ---" % round(time.time() - start_time_all, 2))
