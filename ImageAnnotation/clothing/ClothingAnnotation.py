from fastai.vision import *
import torch
import sys
import pandas as pd
import time
import numpy as np

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager
import warnings; warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

defaults.device = torch.device(config.DEVICE)

class ClothingAnnotator():
    '''
    Rensponsible for color annotation
    '''
    def __init__(self, user, *oids):
        self.user = user
        self.oids = oids
        self.logging = S4F_Logger('ClothingAnnotationLogger', user=user)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)
        self.db_manager = QueryManager(user=self.user)   

        # Create the directories for the product attribute models
        if not os.path.exists(config.PRODUCT_ATTRIBUTE_MODEL_DIR):
            os.makedirs(config.PRODUCT_ATTRIBUTE_MODEL_DIR)
            
        # Initialize Learners (5/5)
        self.necklineLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODEL_NECKLINE)
        self.sleeveLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELSLEEVE)
        self.lengthLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODEL_LENGTH)
        self.collarLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODEL_COLLAR)
        self.fitLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODEL_FIT)

        # Select Products to execute the text based annotation
        filters = config.PRODUCT_ATTRIBUTES + ['Oid', 'ImageSource', 'Image']
        table = 'Product'

        if len(oids) > 0:
            where = ' OR '.join(['Oid=%s' % i for i in  oids])
            filters = '%s' % ', '.join(filters)
            query = 'SELECT %s FROM %s.dbo.%s WHERE %s' % (filters, config.DB_NAME, table, where)
            self.product_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        else:
            params = {attr: 'NULL' for attr in config.ATTRIBUTE_COLUMNS}
            params['table'] = table
            self.product_df = self.db_manager.runSelectQuery(params, filters=filters)


    def execute_annotation(self,):
        start_time = time.time() 
        table = 'Product'
        # Each entry
        self.logger.info("Executing product attribute annotation for %s unlabeled products" % len(self.product_df))
        for index, row in self.product_df.iterrows():
            # check if there is a blob or to skip it
            if not pd.isna(row['Image']):
                image = self.helper.convertBlobToImage(row['Image'])
            else:
                image = self.helper.getWebImage(row['ImageSource'])
            try:
                image = self.helper.convertCVtoPIL(image)
                # Neckline
                if pd.isna(row['NeckDesign']):
                    self.product_df.loc[index, 'NeckDesign'] = self.helper.updateAttribute(config.DICTNECKLINE, image,
                            self.necklineLearner)
                # Sleeve
                if pd.isna(row['Sleeve']):
                    self.product_df.loc[index, 'Sleeve'] = self.helper.updateAttribute(config.DICTSLEEVE, image,
                            self.sleeveLearner)
                # Length
                if pd.isna(row['Length']):
                    self.product_df.loc[index, 'Length'] = self.helper.updateAttribute(config.DICTLENGTH, image,
                            self.lengthLearner)
                # Collar
                if pd.isna(row['CollarDesign']):
                    self.product_df.loc[index, 'CollarDesign'] = self.helper.updateAttribute(config.DICTCOLLAR, image,
                            self.collarLearner)
                # FIT
                if pd.isna(row['Fit']):
                    self.product_df.loc[index, 'Fit'] = self.helper.updateAttribute(config.DICTFIT, image,
                            self.fitLearner)                
                
            except Exception as e:
                self.logger.warn_and_trace(e)
                self.logger.warning('Failed to load image for Product with Oid %s' % row['Oid'])
        # Batch update Product table
        self.logger.info('Updating Product table')
        table = 'Product'
        columns = ['Oid'] + config.PRODUCT_ATTRIBUTES
        self.db_manager.runBatchUpdate(table, self.product_df[columns], 'Oid')

        # End Counting Time
        self.logger.info("--- Finished image annotation of %s records in %s seconds ---" % (len(self.product_df), 
                round(time.time() - start_time, 2)))

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    oids = sys.argv[2:]
    clothing_annotator = ClothingAnnotator(user, *oids)
    clothing_annotator.execute_annotation()