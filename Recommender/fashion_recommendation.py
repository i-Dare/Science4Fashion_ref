import argparse
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD

import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager
from core.helper_functions import *

class FashionRecommender:
    def __init__(self, searchTerm=None, recalc=False, user=config.DEFAULT_USER, threshold=1.):
        
        # Get arguments
        self.threshold = threshold
        self.searchTerm = searchTerm
        self.recalc = recalc
        self.user = user
        self.numberResults = None
        self.searchID = None

        # Logger setup
        self.logging = S4F_Logger('FashionRecommender', user=self.user)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)

        # QueryManager setup
        self.db_manager = QueryManager(user=self.user)

    def textBasedScore(self, products_df):
        '''Calculates recommendation score based on the grades of the user
        '''
        self.logger.info('Calculating recommendation factor according to text similarity')
        # Calculate text features for recommendation 
        products_df, tfidf_vector, vectorizer = self.calculateTextDescriptors(products_df)
        
        # Calculate search term cosine similarity
        preprocessed_searchTerm = self.helper.preprocess_metadata(self.searchTerm)
        query_vec = vectorizer.transform(preprocessed_searchTerm.split())
        # Calculate scemantic score
        products_df['text_score'] = cosine_similarity(tfidf_vector, query_vec).sum(axis=1)
        # Return products with text score > 0
        return products_df[products_df['text_score']>0]

    def orderingBasedScore(self, products_df, trend_mult=1, reference_mult=1):
        '''Calculates recommendation score based on the trending and reference order of the products
           as captured in the 'ProductHistory' table
        '''
        self.logger.info('Calculating recommendation factor according to ordering')
        # Normalize ordering values
        start = time.time()
        scaler = MinMaxScaler((.1, .9))
        ## Calculate ordering score
        trend_factor =  (-trend_mult) * np.log(scaler.fit_transform(products_df['trend_factor'].values.reshape(-1, 1)))
        reference_factor = (-reference_mult) * np.log(scaler.fit_transform(products_df['reference_factor'].values.reshape(-1, 1)))
        products_df['ordering_score'] = trend_factor + reference_factor
        self.logger.info("Calculate ordering score time: {:.2f} ms".format((time.time() - start) * 1000))
        return products_df

    def feedbackBasedScore(self, products_df):
        # check for system's state
        if self.recalc:
            # split data
            seen_rating_ids, irrelevant_ids, unseen_ids = self.split_ids(products_df)
            if unseen_ids.shape[0] == products_df.shape[0]:
                self.logger.warn('No feedback provided.')
                return products_df
            # Predict ratings
            products_df = self.make_prediction(products_df, seen_rating_ids, unseen_ids, action='rating')
            # Predict irrelevance
            products_df = self.make_prediction(products_df, irrelevant_ids, unseen_ids, action='irrelevance')
            # Calculate feedback based score
            # (1-irrelevance) * rating/max(rating)
            products_df['feedback_score'] = (1-products_df['irrelevance']) * (products_df['rating']/products_df['rating'].max())
        else:
            # Remove 'rating' and 'irrelevance' model since this is a new session
            self.helper.remove_model(config.RATING_MODEL)
            self.helper.remove_model(config.IRRELEVANCE_MODEL)
        return products_df   


    def recommendationScore(self, products_df, grading_mult=1, text_mult=1, ordering_mult=1, 
                            color_mult=1, clothing_mult=1):
        '''Calculate recommendation scoring factoring in the relevance of the searchTerms and the 
            relevance/trending ordering
        '''
        scoring_measures = ['text_score', 'feedback_score']
        scoring_factors = [text_mult, grading_mult]
        if not self.recalc:
            scoring_measures = scoring_measures[:2]
            scoring_factors = scoring_factors[:2]

        # Calculated final score
        final_score = np.sum( [mult * products_df[score] for score,mult in zip(scoring_measures, scoring_factors) 
                if score in products_df.columns], axis=0)
        
        products_df['final_score'] = final_score

        # Return items sorted according to final score
        products_df.sort_values('final_score', ascending=False, inplace=True)
        return products_df


    def executeRecommendation(self, ):
        # Get search information
        self.searchID, self.searchTerm, self.numberResults = self.get_search_details()
        # Get all product ranking attributes
        products_df = self.getAttributes()

        # Preprocess ranking attributes
        products_df = self.attributePreprocessing(products_df)

        #
        # Score calculation
        #
        # Calculate recommendation score according to text
        products_df = self.textBasedScore(products_df)
        # Calculate recommendation score according to ordering
        products_df = self.orderingBasedScore(products_df)
        # Calculate recommendation score according to user's feedback
        products_df = self.feedbackBasedScore(products_df)
        # Calculate final recommendation score
        products_df = self.recommendationScore(products_df, text_mult=1, ordering_mult=.5)
        #
        # Register recommendation to the "Result" table if system's state!=recalc, otherwise update 
        # existing recommendation
        self.registerRecommendation(products_df)

        self.evalRecommendation(products_df)

# ----------------------------------------------
#       Utility Functions
# ----------------------------------------------
    def get_search_details(self,):
        # Get search informationn according to the provided search ID
        params = {'table': 'Search', 'Criteria': self.searchTerm, 'CreatedBy': self.user}
        search_df = self.db_manager.runSelectQuery(params)
        search_df.sort_values('Oid', ascending=False, inplace=True)
        searchID = search_df['Oid'].values[0]
        searchTerm = search_df['Criteria'].values[0]
        numberResult = search_df['NumberOfProductsToReturn'].values[0]
        return searchID, searchTerm, numberResult

    def getAttributes(self, ):
        start = time.time()
        # Trigger recalculation
        if self.recalc:
            num_columns = ['ReferenceOrder', 'TrendingOrder', 'Clicked', 'IsFavorite', 'GradeBySystem', 
                'GradeByUser', 'IsIrrelevant']
            # Fetch searchIDs for search term
            params = {'table': 'Search', 'Criteria': self.searchTerm}
            search_df = self.db_manager.runSelectQuery(params)
            searchIDs = search_df['Oid'].tolist()
            # Prepare final query
            where = 'WHERE ' + ' OR '.join([ 'RSLT.Search = %s' % oid for oid in searchIDs])
            query = '''
                SELECT PRD.Oid, PRD.ImageSource, PRD.Description, PRD.Metadata, PRD.ProductTitle,
                        PH.ReferenceOrder, PH.TrendingOrder,
                        RSLT.Search, RSLT.Clicked, RSLT.IsFavorite, RSLT.GradeBySystem, RSLT.GradeByUser, RSLT.IsIrrelevant, RSLT.CreatedBy
                FROM %s.dbo.Product AS PRD 
                    LEFT JOIN %s.dbo.ProductHistory AS PH
                    ON PRD.Oid = PH.Product
                    LEFT JOIN %s.dbo.RESULT AS RSLT
                    ON PRD.oid = RSLT.Product 
                    %s
            ''' % ( config.DB_NAME, config.DB_NAME, config.DB_NAME, where)
        else:
            num_columns = ['ReferenceOrder', 'TrendingOrder']
            query = '''
                        SELECT PRD.Oid, PRD.ImageSource, PRD.Description, PRD.Metadata, PRD.ProductTitle,
                                PH.ReferenceOrder, PH.TrendingOrder
                        FROM %s.dbo.Product AS PRD                          
                            LEFT JOIN %s.dbo.ProductHistory AS PH
                            ON PRD.Oid = PH.Product
                    ''' % (config.DB_NAME, config.DB_NAME)
        products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        # Remove duplicates
        products_df.drop_duplicates(ignore_index=True, inplace=True)
        products_df['Metadata'].fillna('', inplace=True)
        # Max aggregate results per product Oid        
        cat_columns = list(set(products_df.columns) - set(products_df._get_numeric_data().columns.tolist()))
        _products_df = products_df.groupby('Oid').first()[cat_columns]
        _products_df.loc[:, num_columns] = products_df.groupby('Oid').max()[num_columns]

        self.logger.info("Attribute retrieval time: {:.2f} ms".format((time.time() - start) * 1000))
        return _products_df.reset_index()

    def attributePreprocessing(self, products_df):
        '''Sequential attribute preprocessing in stages'''
        # Ordering preprocessing
        products_df = self.orderingPreprocessing(products_df)
        return products_df

    def orderingPreprocessing(self, products_df):
        # Select ordering columns
        columns = products_df.filter(like='Order').columns
        products_df.loc[:, columns] = products_df.loc[:, columns].fillna(value=99999)

        products_df['trend_factor'] = products_df['TrendingOrder']/(products_df['TrendingOrder'] + products_df['ReferenceOrder'])
        products_df['reference_factor'] = products_df['ReferenceOrder']/(products_df['TrendingOrder'] + products_df['ReferenceOrder'])
        return products_df

    def calculateTextDescriptors(self, products_df):
        '''
        Calculates text features for recommendation 
        '''
        # TFIDF vectorizer setup
        start = time.time()
        # If 'recalc' load vectorizer
        if self.recalc:
            vectorizer = self.helper.get_model(config.MODEL_TEXT_DESCRIPTOR)
            tfidf_vector = vectorizer.transform(products_df['Metadata'])
        else:
            vectorizer = TfidfVectorizer(analyzer='word', 
                                        ngram_range=(1,2), 
                                        min_df=3, 
                                        max_df=.95,
            #                              max_features=5000,
                                        use_idf=True, 
                                        stop_words=self.helper.STOP_WORDS)
            tfidf_vector = vectorizer.fit_transform(products_df['Metadata'])
            # saving TFIDF vectorizer
            self.helper.save_model(vectorizer, config.MODEL_TEXT_DESCRIPTOR, config.TEXT_DESCRIPTOR_MODEL_DIR)
        
            # Save TF-IDF vectors
            if not os.path.exists(config.DATADIR):
                os.makedirs(config.DATADIR)
            np.save(config.TFIDF_VECTOR, tfidf_vector.toarray())
        self.logger.info("TFIDF feature extraction time: {:.2f} ms".format((time.time() - start) * 1000))
        return products_df, tfidf_vector, vectorizer

    def split_ids(self, products_df):
        '''Splits data to seen (rated) and useen (unrated)'''
        seen_rating_ids = products_df.loc[(products_df['GradeByUser']>=0) | (products_df['IsIrrelevant']), 'Oid']
        irrelevant_ids = products_df.loc[(products_df['IsIrrelevant']) | (products_df['GradeByUser']>=0), 'Oid']
        
        unseen_ids = products_df.loc[(products_df['GradeByUser']<0) & (~products_df['IsIrrelevant']), 'Oid']
        return seen_rating_ids, irrelevant_ids, unseen_ids
    
    def make_prediction(self, data, train_ids, test_ids, action):
        # Argument assertion
        assert (action in ['rating', 'irrelevance']), 'Action is \"rating\" or \"irrelevance\"'
        
        online_classifiers = {
            'SGD': SGDClassifier(random_state = 42),
            'Perceptron': Perceptron(random_state = 42),
            'NB Multinomial': MultinomialNB(alpha=0.005),
            'Passive-Aggressive': PassiveAggressiveClassifier(C = 0.05, random_state = 42),
            'SVD': MultinomialNB(alpha=0.005)
        }
        
        train_data = data[data['Oid'].isin(train_ids)]
        test_data = data[data['Oid'].isin(test_ids)]
        _, train_x, _ = self.calculateTextDescriptors(train_data)
        _, test_x, _ = self.calculateTextDescriptors(test_data)
        
        if action=='rating':
            # set unratted and irrelevant data to 0
            train_data.loc[train_data['GradeByUser']<0, 'GradeByUser'] = 0
            # set target
            train_y = train_data['GradeByUser'].values
            model_name = config.RATING_MODEL        
            classes = list(range(6))
            # Load or set model
            model = self.helper.get_model(model_name)
            if not model:
                model = online_classifiers['NB Multinomial']
        if action=='irrelevance':
            # set target
            train_y = train_data['IsIrrelevant'].values
            model_name = config.IRRELEVANCE_MODEL
            classes = [True, False]
            # Load or set model
            model = self.helper.get_model(model_name)
            if not model:
                model = online_classifiers['Passive-Aggressive']
        #
        # Train incremental (online) learning model
        #
        # partial training
        model.partial_fit(train_x, train_y, classes=classes)
        # saving model
        self.helper.save_model(model, model_name, config.INCREMENTAL_LEARNING_MODEL_DIR)
        # predition
        prediction = model.predict(test_x)
        # Save prediction to products dataframe
        if action=='rating':
            data.loc[data['Oid'].isin(train_ids), 'rating'] = train_y.tolist()
            data.loc[data['Oid'].isin(test_ids), 'rating'] = prediction.tolist()
        if action=='irrelevance':
            data.loc[data['Oid'].isin(train_ids), 'irrelevance'] = train_y.tolist()
            data.loc[data['Oid'].isin(test_ids), 'irrelevance'] = prediction.tolist()
        return data

    def registerRecommendation(self, products_df):
        '''Stores the recommendation as a new record in Results table'''
        # During recalculation (state!=0) remove previous results according to Search Oid
        if self.recalc:
            for i, row in products_df.iterrows():
                uniq_params = {'table': 'Result', 
                        'Product': row['Oid'],
                        'Search': self.searchID}
                params = {'table': 'Result', 
                        'Search': self.searchID,
                        'Product': row['Oid'],
                        'IsIrrelevant': row['IsIrrelevant'],     
                        'GradeByUser': row['GradeByUser'],                    
                        'GradeBySystem': row['final_score']}
                self.db_manager.runCriteriaUpdateQuery(uniq_params=uniq_params, params=params)

            # query = "DELETE FROM %s.dbo.Result  WHERE Search=%s" % (config.DB_NAME, self.searchID)
            # self.db_manager.runSimpleQuery(query)
        else:
            for i, row in products_df.iterrows():               
                params = {'table': 'Result', 
                'Search': self.searchID,
                'Product': row['Oid'],
                'GradeBySystem': row['final_score']}
                self.db_manager.runInsertQuery(params, get_identity=True)

# ----------------------------------------------
#       Evaluation Functions
# ----------------------------------------------
    def evalRecommendation(self, recommendation_df):
        where = ' OR '.join(['Oid=%s' % oid for oid in recommendation_df['Oid'].values])
        query = ''' 
                SELECT Oid, Metadata, ProductTitle, Description, ImageSource, Image  FROM %s.dbo.Product
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
            self.logger.info('%s: %s - %s\n%s' % (oid, score, text, imgUrl))



if __name__ == "__main__":
    #
    # Initialize argument parser
    #
    parser = argparse.ArgumentParser(description = 'A script for executing the recommendation \
            functionality', prog = 'FashionRecommender')
    parser.add_argument('-s','--searchTerm', type = str, help = '''Input the search terms of the query''', 
            required = True, nargs = '?')
    parser.add_argument('-u', '--user', default=config.DEFAULT_USER, type = str, help = '''Input user''')
    parser.add_argument('-r', '--recalc', help = '''Triggers the recalculation functionality \
            of the FashionRecommender, where a new set of recommendations is generated after user's \
            feedback''', action="store_true", default = False)

    # Parse arguments
    args, unknown = parser.parse_known_args()

    # Get arguments
    searchTerm = args.searchTerm
    recalc = args.recalc
    user = args.user

    recommender = FashionRecommender(searchTerm=searchTerm, 
                                     recalc=recalc, 
                                     user=user)
    recommender.executeRecommendation()     
