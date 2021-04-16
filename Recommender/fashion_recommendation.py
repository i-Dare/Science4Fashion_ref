import argparse
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime

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

        # QueryManager setup
        self.db_manager = QueryManager(user=self.user)

    def getRatedItems(self,):
        return pd.DataFrame()
    
    def textBasedRecommendation(self,):
        return pd.DataFrame()

    def executeRecommendation(self, ):
        # Initialize recommender by retrieving items rated by the user
        rated_items_df = self.getRatedItems()

        # No items are rated (cold start) perform text based retrieval on all items
        if rated_items_df.empty:
            recommendation_df = self.textBasedRecommendation()





if __name__ == "__main__":
   recommender = FashionRecommender()
   recommender.executeRecommendation()        