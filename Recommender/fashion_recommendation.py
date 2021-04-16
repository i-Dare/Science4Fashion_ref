import argparse
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity 

import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager
from core.helper_functions import *

class FashionRecommender:
    def __init__(self,):
        #
        # Initialize argument parser
        #
        self.parser = argparse.ArgumentParser(description = 'A script for executing the recommendation \
            functionality', prog = 'FashionRecommender')
        self.parser.add_argument('-s','--searchTerm', type = str, help = '''Input the search query term''', 
                required = True, nargs = '+')
        self.parser.add_argument('-n','--numberResults', type = int, help = '''Input the number of \
                results you want to return''', default = 10, nargs = '?')
        self.parser.add_argument('-u', '--user', default=config.DEFAULT_USER, type = str, help = '''Input user''')

        # Parse arguments
        self.args, unknown = self.parser.parse_known_args()

        # Get arguments
        self.searchTerm = ' '.join(self.args.searchTerm)
        self.numberResults = self.args.numberResults

        # Logger setup
        self.user = self.args.user
        self.logging = S4F_Logger('FashionRecommender', user=self.user)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)

        # QueryManager setup
        self.db_manager = QueryManager(user=self.user)

    def getRatedItems(self,):
        user_rating_df = pd.read_sql_query('''SELECT PRD.Oid, RSLT.Clicked, \
                                                     RSLT.IsFavorite, RSLT.GradeBySystem, \
                                                     RSLT.GradeByUser, RSLT.CreatedBy
                                            FROM %s.dbo.PRODUCT AS PRD
                                            LEFT JOIN %s.dbo.RESULT AS RSLT
                                            ON PRD.oid = RSLT.Product
                                            WHERE RSLT.CreatedBy=\'%s\'''' % \
                                            (config.DB_NAME, config.DB_NAME, self.user), config.ENGINE)
        return user_rating_df
    
    def textBasedRecommendation(self,):
        # # Load TFID embedings
        # tfidf_vector = np.load(config.TFIDF_VECTOR)
        # # Load TFIDF vectorizer
        # vectorizer = self.helper.get_model(config.MODEL_TEXT_DESCRIPTOR)

        # Calculate text features for recommendation 
        product_df, tfidf_vector, vectorizer = self.helper.calculateTextDescriptors()
        
        # Calculate search term cosine similarity
        preprocessed_searchTerm = self.helper.preprocess_metadata(self.searchTerm)
        query_vec = vectorizer.transform(preprocessed_searchTerm.split())
        product_df['cosine_score'] = cosine_similarity(tfidf_vector, query_vec).sum(axis=1)
        
        # Return sorted according to cosine similarity
        product_df.sort_values('cosine_score', ascending=False, inplace=True)
        return product_df.iloc[:self.numberResults]


    def executeRecommendation(self, ):
        # Initialize recommender by retrieving items rated by the user
        user_rating_df = self.getRatedItems()

        # No items are rated (cold start) perform text based retrieval on all items
        if user_rating_df.empty:
            recommendation_df = self.textBasedRecommendation()
        self.evalRecommendation(recommendation_df)


    def evalRecommendation(self, recommendation_df):
        where = ' OR '.join(['Oid=%s' % oid for oid in recommendation_df['Oid'].values])
        query = ''' 
                SELECT Oid, Metadata, ProductTitle, Description, ImageSource  FROM %s.dbo.Product
                WHERE %s
                ''' % (config.DB_NAME, where)
        product_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        product_df['combined_columns'] = product_df['Metadata'].astype(str) \
                                        + product_df['ProductTitle'].astype(str) \
                                        + product_df['Description'].astype(str)
        for i, row in recommendation_df.iterrows():
            oid = row['Oid']
            score = row['cosine_score']
            text = product_df.loc[product_df['Oid']==oid, 'combined_columns'].values[0]
            imgUrl = product_df.loc[product_df['Oid']==oid, 'ImageSource'].values[0]
            print('%s: %s - %s\n%s' % (oid, score, text, imgUrl))




if __name__ == "__main__":
   recommender = FashionRecommender()
   recommender.executeRecommendation()        