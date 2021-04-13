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

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    logging = S4F_Logger('ClothingAnnotationLogger', user=user)
    logger = logging.logger
    helper = Helper(logging)
    db_manager = QueryManager(user=user)

    # Set Device
    defaults.device = torch.device(config.DEVICE)

    # Create the directories for the product attribute models
    if not os.path.exists(config.PRODUCT_ATTRIBUTE_MODEL_DIR):
        os.makedirs(config.PRODUCT_ATTRIBUTE_MODEL_DIR)
        
    # Initialize Learners (5/5)
    necklineLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELNECKLINE)
    sleeveLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELSLEEVE)
    lengthLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELLENGTH)
    collarLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELCOLLAR)
    fitLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELFIT)

    params = {attr: 'NULL' for attr in config.ATTRIBUTE_COLUMNS}
    params['table'] = 'Product'
    product_df = db_manager.runSelectQuery(params)

    # Each entry
    logger.info("Executing product attribute annotation for %s unlabeled products" % len(product_df))
    for index, row in product_df.iterrows():
        # check if there is a blob or to skip it
        if row['Image'] is not None:
            image = helper.convertBlobToImage(row['Image'])
            image = helper.convertCVtoPIL(image)
            # Neckline
            if row['NeckDesign'] == None:
                product_df.loc[index, 'NeckDesign'] = helper.updateAttribute(config.DICTNECKLINE, image,
                        necklineLearner)
            # Sleeve
            if row['Sleeve'] == None:
                product_df.loc[index, 'Sleeve'] = helper.updateAttribute(config.DICTSLEEVE, image,
                        sleeveLearner)
            # Length
            if row['Length'] == None:
                product_df.loc[index, 'Length'] = helper.updateAttribute(config.DICTLENGTH, image,
                        lengthLearner)
            # Collar
            if row['CollarDesign'] == None:
                product_df.loc[index, 'CollarDesign'] = helper.updateAttribute(config.DICTCOLLAR, image,
                        collarLearner)
            # FIT
            if row['Fit'] == None:
                product_df.loc[index, 'Fit'] = helper.updateAttribute(config.DICTFIT, image,
                        fitLearner)
            
            # Update Product table
            uniq_params = {'table': 'Product', 'Oid': row['Oid']}
            params = {attr: product_df.loc[index, attr] for attr in config.ATTRIBUTE_COLUMNS}
            params['table'] = 'Product'
            _ = db_manager.runCriteriaUpdateQuery(uniq_params=uniq_params, params=params)
        else:
            logger.warning('Failed to load image for Product with Oid %s' % row['Oid'])

    # End Counting Time
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    logger.info("Updated %s records in %s seconds ---" % (len(product_df), round(time.time() - start_time, 2)))