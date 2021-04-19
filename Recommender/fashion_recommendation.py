import argparse
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

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

    def gradeBasedScore(self, products_df):
        '''Calculates recommendation score based on the cosine similarity of the TFIDF embedings of
           the processed text attributes and the processed search term
        '''
        self.logger.info('Calculating recommendation factor according to user\'s rating')
        return products_df

    def textBasedScore(self, products_df):
        '''Calculates recommendation score based on the grades of the user
        '''
        self.logger.info('Calculating recommendation factor according to text similarity')
        # Calculate text features for recommendation 
        products_df, tfidf_vector, vectorizer = self.calculateTextDescriptors(products_df)
        
        # Calculate search term cosine similarity
        preprocessed_searchTerm = self.helper.preprocess_metadata(self.searchTerm)
        query_vec = vectorizer.transform(preprocessed_searchTerm.split())
        # Calculate ordering score
        products_df['text_score'] = cosine_similarity(tfidf_vector, query_vec).sum(axis=1)        
        return products_df

    def orderingBasedScore(self, products_df, trend_mult=1, reference_mult=1):
        '''Calculates recommendation score based on the trending and reference order of the products
           as captured in the 'ProductHistory' table
        '''
        self.logger.info('Calculating recommendation factor according to ordering')
        # Get trend/reference order
        start = time.time()

        print("Fetching ordering time: {:.2f} ms".format((time.time() - start) * 1000))

        # Normalize ordering values
        start = time.time()
        scaler = MinMaxScaler((0.1, .5))
        # Calculate ordering score
        trend_factor = trend_mult * products_df['trend_factor']
        reference_factor = reference_mult * products_df['reference_factor']
        products_df['ordering_score'] = scaler.fit_transform((2*(trend_factor * reference_factor)/(trend_factor + reference_factor)).values.reshape(-1, 1))
        print("Calculate ordering score time: {:.2f} ms".format((time.time() - start) * 1000))
        return products_df

    def recommendationScore(self, products_df, grading_mult=1, text_mult=1, ordering_mult=1, 
                            color_mult=1, clothing_mult=1):
        '''Calculate recommendation scoring
        '''
        scoring_measures = ['grading_score', 'text_score', 'ordering_score', 'color_score', 
                'clothing_score']

        # Calculated final score
        final_score = np.sum( [mult * products_df[score] for score,mult in zip(scoring_measures, 
                [grading_mult, text_mult, ordering_mult, color_mult, clothing_mult]) 
                if score in products_df.columns], axis=0)
        
        products_df['final_score'] = final_score

        # Return items sorted according to final score
        products_df.sort_values('final_score', ascending=False, inplace=True)
        return products_df.iloc[:self.numberResults]


    def executeRecommendation(self, ):
        # Get all product ranking attributes
        self.all_products_df = self.getAttributes()

        # Preprocess ranking attributes
        products_df = self.attributePreprocessing()
        # Calculate recommendation score according to text
        products_df = self.textBasedScore(products_df)
        # Calculate recommendation score according to ordering
        products_df = self.orderingBasedScore(products_df)

        #
        # Calculate final recommendation score
        #
        products_df = self.recommendationScore(products_df)
        
        self.evalRecommendation(products_df)

# ----------------------------------------------
#       Utility Functions
# ----------------------------------------------
    def getAttributes(self, ):
        start = time.time()
        attJoin, attSelect = '', []
        for i, attr in enumerate(config.PRODUCT_ATTRIBUTES):
            attSelect += ['attr%s.Description AS %s' % (i, attr)]
            attJoin += ' LEFT JOIN %s.dbo.%s AS attr%s\nON PRD.%s = attr%s.Oid ' \
                     % (config.DB_NAME, attr, i, attr, i)

        query = '''
                    SELECT PRD.Oid, PRD.ImageSource, PRD.Description, PRD.Metadata, PRD.ProductTitle,
                           PH.ReferenceOrder, PH.TrendingOrder,
                           PC.Ranking,                   
                           C.Label, C.LabelDetailed, C.Red, C.Blue, C.Green,
                           RSLT.Clicked, RSLT.IsFavorite, RSLT.GradeBySystem, RSLT.GradeByUser, RSLT.CreatedBy,
                           %s
                    FROM %s.dbo.ProductColor AS PC
                        LEFT JOIN %s.dbo.ColorRGB AS C
                        ON PC.ColorRGB = C.Oid
                        LEFT JOIN %s.dbo.Product AS PRD
                        ON PC.Product=PRD.Oid
                        LEFT JOIN %s.dbo.ProductHistory AS PH
                        ON PRD.Oid = PH.Product
                        LEFT JOIN %s.dbo.RESULT AS RSLT
                        ON PRD.oid = RSLT.Product 
                        %s
                    WHERE RSLT.CreatedBy= \'%s\'      
                ''' % (','.join(attSelect), config.DB_NAME, config.DB_NAME, config.DB_NAME, 
                config.DB_NAME, config.DB_NAME, attJoin, self.user)

        products_df = self.db_manager.runSimpleQuery(query, get_identity=True)                                        

        # No items are rated (cold start) perform text based retrieval on all items
        if products_df.empty:
            query = '''
                        SELECT PRD.Oid, PRD.ImageSource, PRD.Description, PRD.Metadata, PRD.ProductTitle,
                               PH.ReferenceOrder, PH.TrendingOrder,
                               PC.Ranking,                   
                               C.Label, C.LabelDetailed, C.Red, C.Blue, C.Green,
                               %s
                        FROM %s.dbo.ProductColor AS PC
                            LEFT JOIN %s.dbo.ColorRGB AS C
                            ON PC.ColorRGB = C.Oid
                            LEFT JOIN %s.dbo.Product AS PRD
                            ON PC.Product=PRD.Oid
                            LEFT JOIN %s.dbo.ProductHistory AS PH
                            ON PRD.Oid = PH.Product
                            %s
                    ''' % (','.join(attSelect), config.DB_NAME, config.DB_NAME, config.DB_NAME, 
                    config.DB_NAME, attJoin)
            products_df = self.db_manager.runSimpleQuery(query, get_identity=True)

        print("Attribute retrieval time: {:.2f} ms".format((time.time() - start) * 1000))
        return products_df

    def attributePreprocessing(self,):
        # Group by products
        products_df = self.all_products_df.groupby('Oid').mean().reset_index()
        # Text preprocessing
        products_df = self.textPreprocessing(products_df)
        # Ordering preprocessing
        products_df = self.orderingPreprocessing(products_df)
        return products_df

    def textPreprocessing(self, products_df):
        # Combine text fields
        _all_products_df = self.all_products_df.copy()
        start = time.time()
        for attr in config.PRODUCT_ATTRIBUTES:
            _all_products_df[attr] = _all_products_df[attr].str.replace(' ', '_')

        text_columns = config.PRODUCT_ATTRIBUTES + ['Metadata', 'ProductTitle', 'Description', 'Label', 'LabelDetailed']
        
        products_df['combined_text'] = (_all_products_df.groupby('Oid')
                .apply( lambda row:  [list(set(row[col])) for col in text_columns ] )
                .apply(lambda row:  [r for r in (sum(row, []))  if r is not None])
                .str.join(' ')).values

        products_df.loc[:, 'processed_combined_text'] = products_df.loc[:, 'combined_text'].apply(self.helper.preprocess_metadata)
        print("Text preprocessing time: {:.2f} ms".format((time.time() - start) * 1000))
        return products_df

    def orderingPreprocessing(self, products_df):
        # Select ordering columns
        columns = products_df.filter(like='Order').columns
        for col in columns:
            products_df.loc[:, col] = products_df[col].fillna(value=99999)

        scaler = MinMaxScaler((0.1, 1.))
        products_df['trend_factor'] = 1/scaler.fit_transform(products_df['TrendingOrder'].values.reshape(-1,1))
        products_df['reference_factor'] = 1/scaler.fit_transform(products_df['ReferenceOrder'].values.reshape(-1,1))
        return products_df

    def calculateTextDescriptors(self, products_df):
        '''
        Calculates text features for recommendation 
        '''
        # TFIDF vectorizer setup
        start = time.time()
        vectorizer = TfidfVectorizer(analyzer='word', 
                                    ngram_range=(1,2), 
                                    min_df=3, 
                                    max_df=.95,
        #                              max_features=5000,
                                    use_idf=True, 
                                    stop_words=self.helper.STOP_WORDS)
        # saving TFIDF features
        tfidf_vector = vectorizer.fit_transform(products_df['processed_combined_text'])
        print("TFIDF feature extraction time: {:.2f} ms".format((time.time() - start) * 1000))

        if not os.path.exists(config.DATADIR):
            os.makedirs(config.DATADIR)
        np.save(config.TFIDF_VECTOR, tfidf_vector)
        # saving TFIDF vectorizer
        self.helper.save_model(vectorizer, config.MODEL_TEXT_DESCRIPTOR, config.TEXT_DESCRIPTOR_MODEL_DIR)
        return products_df, tfidf_vector, vectorizer


    def evalRecommendation(self, recommendation_df):
        where = ' OR '.join(['Oid=%s' % oid for oid in recommendation_df['Oid'].values])
        query = ''' 
                SELECT Oid, Metadata, ProductTitle, Description, ImageSource  FROM %s.dbo.Product
                WHERE %s
                ''' % (config.DB_NAME, where)
        products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        products_df['combined_columns'] = products_df['Metadata'].astype(str) \
                                        + products_df['ProductTitle'].astype(str) \
                                        + products_df['Description'].astype(str)
        for _, row in recommendation_df.iterrows():
            oid = row['Oid']
            score = row['text_score']
            text = products_df.loc[products_df['Oid']==oid, 'combined_columns'].values[0]
            imgUrl = products_df.loc[products_df['Oid']==oid, 'ImageSource'].values[0].replace('\\','')
            print('%s: %s - %s\n%s' % (oid, score, text, imgUrl))




if __name__ == "__main__":
   recommender = FashionRecommender()
   recommender.executeRecommendation()        