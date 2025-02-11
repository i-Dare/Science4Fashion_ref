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

MODEL_DICT = {
    'NeckDesign': config.MODEL_NECKLINE, 
    'Sleeve': config.MODEL_SLEEVE, 
    'Length': config.MODEL_LENGTH, 
    'CollarDesign': config.MODEL_COLLAR, 
    'Fit': config.MODEL_FIT
}
LABEL_DICT = {
    'NeckDesign': config.DICTNECKLINE, 
    'Sleeve': config.DICTSLEEVE, 
    'Length': config.DICTLENGTH, 
    'CollarDesign': config.DICTCOLLAR, 
    'Fit': config.DICTFIT
}


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
            
        # Select Products to execute the text based annotation
        filters = config.ATTRIBUTE_COLUMNS + ['Oid', 'ImageSource', 'Image', 'Sketch', 'Photo']
        table = 'Product'

        if len(oids) > 0:
            where = ' OR '.join(['Oid=%s' % i for i in  oids])
            filters = '%s' % ', '.join(filters)
            query = 'SELECT %s FROM %s.dbo.%s WHERE %s' % (filters, config.DB_NAME, table, where)
            self.products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        else:
            params = {attr: 'NULL' for attr in config.ATTRIBUTE_COLUMNS}
            params['table'] = table

            attSelect = ' ,'.join(map('PRD.{0}'.format, filters)) 
            where = ' AND '.join(['PRD.%s is NULL' % attr for attr in  config.ATTRIBUTE_COLUMNS])
            query = '''SELECT %s
                FROM %s.dbo.%s AS PRD 
                WHERE %s''' % (attSelect, config.DB_NAME, table, where)

            self.products_df = self.db_manager.runSimpleQuery(query, get_identity=True)

    def annotate(self, modelName, image, productID):
        # Initialize Learner
        model = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, MODEL_DICT[modelName])
        labels = LABEL_DICT[modelName]
        # Annotate
        try:
            product = self.helper.updateAttribute(labels, image, model)
        except Exception as e:
            self.logger.warn_and_trace(e, extra={'Product': productID})
            self.logger.warning('Failed to infer %s for Product with Oid %s' % (modelName, productID), 
                    extra={'Product': productID})
        return product

    def execute_annotation(self,):
        start_time = time.time() 
        table = 'Product'
        cnt = 0

        self.logger.info("Executing product attribute annotation for %s unlabeled product(s)" % len(self.products_df))
        # Batch iteration
        step = config.BATCH_STEP
        for i in self.products_df.index[::step]:
            chunk = self.products_df.loc[self.products_df.index[i:i+step]]
            for row in chunk.itertuples():
                productID = row.Oid
                self.logger.debug('Clothing annotation for product %s' % productID, extra={'Product': productID})
                image = self.helper.imageExtraction(row._asdict())          
                if image is not None:
                    image = self.helper.convertCVtoPIL(image)
                    for modelName in ['NeckDesign', 'Sleeve', 'Length', 'CollarDesign', 'Fit']:
                        try:
                            chunk.loc[row.Index, modelName] = self.annotate(modelName, image, productID)
                        except Exception as e:
                            self.logger.warning('Failed to extract %s informantion for product %s' % 
                                    (modelName, productID), extra={'Product': productID})
                else:
                    cnt += 1
                    self.logger.warning('Failed to extract image for product %s' % productID, 
                            extra={'Product': productID})
            # Batch update Product table
            self.logger.info('Updating Product table after product attribute annotation')
            table = 'Product'
            columns = ['Oid'] + config.ATTRIBUTE_COLUMNS
            self.db_manager.runBatchUpdate(table, chunk[columns], 'Oid')


        # for row in self.products_df.itertuples():
        #     productID = row.Oid
        #     self.logger.debug('Clothing annotation for product %s' % productID, extra={'Product': productID})
        #     image = self.helper.imageExtraction(row._asdict())          
        #     if image is not None:
        #         image = self.helper.convertCVtoPIL(image)
        #         for modelName in ['NeckDesign', 'Sleeve', 'Length', 'CollarDesign', 'Fit']:
        #             try:
        #                 self.products_df.loc[row.Index, modelName] = self.annotate(modelName, image, productID)
        #             except Exception as e:
        #                 self.logger.warning('Failed to extract %s informantion for product %s' % 
        #                         (modelName, productID), extra={'Product': productID})
        #     else:
        #         cnt += 1
        #         self.logger.warning('Failed to extract image for product %s' % productID, 
        #                 extra={'Product': productID})


        # # Batch update Product table
        # self.logger.info('Updating Product table after product attribute annotation')
        # table = 'Product'
        # columns = ['Oid'] + config.ATTRIBUTE_COLUMNS
        # self.db_manager.runBatchUpdate(table, self.products_df[columns], 'Oid')

        # End Counting Time
        self.logger.info("--- Finished product attribute annotation of %s records in %s seconds ---" 
                % (len(self.products_df) - cnt, round(time.time() - start_time, 2)))
        self.logger.close()

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    oids = sys.argv[2:-1]
    loglevel = sys.argv[-1]
    clothing_annotator = ClothingAnnotator(user, *oids, loglevel=loglevel)
    clothing_annotator.execute_annotation()