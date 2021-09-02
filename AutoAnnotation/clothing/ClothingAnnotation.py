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
    def __init__(self, user, *oids, loglevel=config.DEFAULT_LOGGING_LEVEL):
        self.user = user
        self.oids = oids
        self.loglevel = loglevel
        self.logging = S4F_Logger('ClothingAnnotationLogger', user=user, level=self.loglevel)
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
            self.products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        else:
            params = {attr: 'NULL' for attr in config.ATTRIBUTE_COLUMNS}
            params['table'] = table
            self.products_df = self.db_manager.runSelectQuery(params, filters=filters)


    def execute_annotation(self,):
        start_time = time.time() 
        table = 'Product'
        # Each entry
        self.logger.info("Executing product attribute annotation for %s unlabeled products" % len(self.products_df))
        for index, row in self.products_df.iterrows():
            productID = row['Oid']
            self.logger.debug('Clothing annotation for product %s' % productID, extra={'Product': productID})
            # check if there is a blob or to skip it
            if not pd.isna(row['Image']):
                image = self.helper.convertBlobToImage(row['Image'])
            else:
                image = self.helper.getWebImage(row['ImageSource'])
            image = self.helper.convertCVtoPIL(image)
            # Neckline
            try:
                if pd.isna(row['NeckDesign']):
                    self.products_df.loc[index, 'NeckDesign'] = self.helper.updateAttribute(config.DICTNECKLINE, image,
                            self.necklineLearner)
            except Exception as e:
                self.logger.warn_and_trace(e, extra={'Product': productID})
                self.logger.warning('Failed to infer Neckline for Product with Oid %s' % productID, extra={'Product': productID})

            # Sleeve
            try:                
                if pd.isna(row['Sleeve']):
                    self.products_df.loc[index, 'Sleeve'] = self.helper.updateAttribute(config.DICTSLEEVE, image,
                            self.sleeveLearner)

            except Exception as e:
                self.logger.warn_and_trace(e, extra={'Product': productID})
                self.logger.warning('Failed to infer Sleeve for for Product with Oid %s' % productID, extra={'Product': productID})

            # Length
            try:  
                if pd.isna(row['Length']):
                    self.products_df.loc[index, 'Length'] = self.helper.updateAttribute(config.DICTLENGTH, image,
                            self.lengthLearner)

            except Exception as e:
                self.logger.warn_and_trace(e, extra={'Product': productID})
                self.logger.warning('Failed to infer Length for for Product with Oid %s' % productID, extra={'Product': productID})

            # Collar
            try:  
                if pd.isna(row['CollarDesign']):
                    self.products_df.loc[index, 'CollarDesign'] = self.helper.updateAttribute(config.DICTCOLLAR, image,
                            self.collarLearner)

            except Exception as e:
                self.logger.warn_and_trace(e, extra={'Product': productID})
                self.logger.warning('Failed to infer Collar for for Product with Oid %s' % productID, extra={'Product': productID})

            # Fit
            try:  
                if pd.isna(row['Fit']):
                    self.products_df.loc[index, 'Fit'] = self.helper.updateAttribute(config.DICTFIT, image,
                            self.fitLearner)                
                
            except Exception as e:
                self.logger.warn_and_trace(e, extra={'Product': productID})
                self.logger.warning('Failed to infer Fit for for Product with Oid %s' % productID, extra={'Product': productID})

        # Batch update Product table
        self.logger.info('Updating Product table after product attribute annotation')
        table = 'Product'
        columns = ['Oid'] + config.PRODUCT_ATTRIBUTES
        self.db_manager.runBatchUpdate(table, self.products_df[columns], 'Oid')

        # End Counting Time
        self.logger.info("--- Finished product attribute annotation of %s records in %s seconds ---" % (len(self.products_df), 
                round(time.time() - start_time, 2)))
        self.logger.close()

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    oids = sys.argv[2:-1]
    loglevel = sys.argv[-1]
    clothing_annotator = ClothingAnnotator(user, *oids, loglevel=loglevel)
    clothing_annotator.execute_annotation()