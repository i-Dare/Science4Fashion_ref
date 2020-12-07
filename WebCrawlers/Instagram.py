#Testing Instagram Crawler
#Medium Blog useful
#https://medium.com/@adamaulia/crawling-instagram-using-instalooter-2791edb453ff
#Documentation
#https://instalooter.readthedocs.io/en/latest/instalooter/index.html

#from instalooter.looters import ProfileLooter
#looter = ProfileLooter('eilex_kyp')
from instalooter.looters import HashtagLooter
import os
import json
from urllib.request import urlretrieve
from datetime import datetime
import regex as re
# import constants
import sqlalchemy
import pandas as pd
from textblob import TextBlob
import helper_functions
from instaloader import Instaloader
from instaloader import Hashtag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import config



def get_download_path(index, imgsource, hashtag):
  path = os.getcwd()
  path = path + '\\' + 'InstagramImages\\' + hashtag + '\\'
  if not os.path.isdir(path):
    os.makedirs(path)
  now = datetime.now()
  dt_string = now.strftime("%Y/%m/%d+%H:%M:%S")  # get current datetime
  image_path = path + 'Instagram' + str(index) + "_" + re.sub("[ /:]", ".", dt_string) + '.jpeg'
  try:
      urlretrieve(imgsource, image_path)
      return image_path
  except TypeError:
      pass


def convertToBinaryData(filename):
  # Convert digital data to binary format
  with open(filename, 'rb') as file:
      blobData = file.read()
  return blobData


def r(insta):
    # Load custom attributes from fashio word list and custom attributes used in the image annotation module:
    # Fashion attributes
    file_path = os.path.join(helper_functions.WEB_CRAWLERS, 'PinterestCrawler', 'fashion_words_700.txt')
    fashion_att_file = open(file_path, "r")
    fashion_att = fashion_att_file.read().split(',')
    fashion_att_file.close()
    # Image annotation labels
    fashionLabels = pd.read_excel(config.NRGATTRIBUTESPATH, sheet_name=config.SHEETNAME)
    attributList = np.hstack(
        [fashionLabels[attr].replace(' ', np.nan).dropna().unique() for attr in config.NRGATTRIBUTES]).tolist()
    fashion_att = helper_functions.preprocess_words(fashion_att + attributList)

    instaDF = pd.DataFrame(insta)
    # Preprocess metadata
    instaDF['processed_metadata'] = instaDF['description'].apply(helper_functions.preprocess_metadata)
    # Preprocess query
    instaDF['query'] = instaDF['query'].apply(lambda row: ' '.join(helper_functions.preprocess_words(row.split())))

    ## Calculate a factor for tokens that appear in metatdata
    keywords = instaDF[0]['query']
    instaDF['factor'] = instaDF['processed_metadata'].apply(
        lambda row: len(set([word for word in row if word in keywords])) / len(keywords))

    ## Calculate a factor based on the cosine similarity of TFIDF transformation of the query terms and
    # the processed metadata using the fashion expert terminology as vocabulary
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
    vectorizer.fit_transform(fashion_att)
    tfidf_vector = vectorizer.transform(instaDF['processed_metadata'])
    query_vector = vectorizer.transform(instaDF['query'])

    ## Calculate cosine similarity
    cosine_vector = cosine_similarity(query_vector[0], tfidf_vector)
    instaDF['cosine ranking score'] = np.hstack(cosine_vector).tolist() * instaDF['factor']

    ## Calculate a factor based on Pinterest's recommendation (order of result parsing)
    scaler = MinMaxScaler()
    pinterest_score = scaler.fit_transform(np.arange(len(instaDF)).reshape(-1, 1))
    instaDF.loc[instaDF.sort_values(by='timestamp', ascending=False).index, 'pinterest score'] = pinterest_score

    ## Calculate Final Ranking Score giving the cosine similarity factor a greater score than the
    # factor based on the Pinterest recommendation
    instaDF['final score'] = (instaDF['cosine ranking score'] * 0.7) + (instaDF['pinterest score'] * 0.3)
    instaDF.sort_values(by='final score', ascending=False, inplace=True)

    # Save ranked results to the database
    for _, row in instaDF.iterrows():
        site = 'Instagram'
        searchwords = ''.join(keywords.split())
        imageFilePath = row['imageFilePath']
        url = row['URL']
        imgURL = row['imgURL']
        empPhoto = helper_functions.getImage(imgURL, imageFilePath)
        head = row['title']
        meta = row['description']
        helper_functions.addNewProduct(site,
                                       searchwords,
                                       imageFilePath,
                                       empPhoto,
                                       url,
                                       imgURL,
                                       head,
                                       None,
                                       None,
                                       None,
                                       meta,
                                       None,
                                       None)



if __name__ == '__main__':
    ########################################### SEARCH PATH KEYWORDS ###########################################
    currendDir = helper_functions.WEB_CRAWLERS
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME

    ########################################### Open the file with read only permit ###########################################
    file = open(os.path.join(currendDir, 'keywords.txt'), "r")

    ########################################### Use readlines to read all lines in the file ###########################################
    lines = file.readlines()  # The variable "lines" is a list containing all lines in the file
    file.close()  # Close the file after reading the lines.


    ########################################### SCRAPE IMAGES FOR EVERY ENTRY IN KEYWORDS ###########################################
    print(len(lines))
    for i in range(0, len(lines)):
        keys = lines[i]
        keys = keys.replace('\n', '')
        print("Crawler Search no." + str(i + 1) + ' ------------------- Search query: "' + str(keys) + '"')  #

        keywords = keys.split(" ")
        keyLen = len(keywords)
        keyUrl = keywords[1].strip('"')
        breakNumber = int(keywords[0])
        for j in range(2, keyLen):
            keyUrl = keyUrl + ' ' + keywords[j].strip('"')

        print('Query: ' + str(keyUrl))
        print("Number of crawled images wanted: " + str(breakNumber))

        ########################################### Scraper / Hashtag ###########################################
        search = keyUrl.replace(' ','')
        hashtagtext = search.replace('-','')

        threshold = breakNumber
        ASK_SQL_Query = pd.read_sql_query('''SELECT * FROM S4F.dbo.Product''', engine)
        sqlsocial = pd.DataFrame(ASK_SQL_Query)
        L = Instaloader()
        L.login(config.INSTAGRAM_USERNAME, config.INSTAGRAM_PASSWORD)
        hashtag = Hashtag.from_name(L.context, hashtagtext)
        count = 0;
        insta = []

        for post in hashtag.get_posts():
            count = count + 1;
            post_url = "https://www.instagram.com/p/" + str(post.shortcode) + "/"
            imgsource = post.url
            testdf = sqlsocial.loc[sqlsocial['URL'] == post_url]
            video = post.is_video
            # print(helper_functions.setImageFilePath(post_url, hashtagtext,count))
            imageFilePath = helper_functions.setImageFilePath(post_url, hashtagtext,count)
            if testdf.empty and not video:
                post_info = " ".join(re.findall("[a-zA-Z]+", post.caption))
                post_hashtags = post.caption_hashtags
                post_likes = post.likes
                post_date = post.date
                insta.append(({'query': hashtagtext,
                               'timestamp': post_date,
                               'url': post_url,
                               'imgURL': imgsource,
                               'imageFilePath':imageFilePath,
                               'title': None,
                               'description': post_info}))
            if count > threshold:
                break
        r(insta)





