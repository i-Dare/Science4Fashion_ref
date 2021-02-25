from fastai.vision import *
import torch
import sys
import pandas as pd
import time
import numpy as np

from helper_functions import *
import config
from logger import S4F_Logger
import warnings; warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    logfile = 'tmp.log'
    logger = S4F_Logger('ClothingAnnotationLogger', logfile=logfile).logger
    helper = Helper(logger)

    # Set Device
    defaults.device = torch.device(config.DEVICE)

    # Initialize Learners (5/5)
    necklineLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELNECKLINE)
    sleeveLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELSLEEVE)
    lengthLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELLENGTH)
    collarLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELCOLLAR)
    fitLearner = load_learner(config.PRODUCT_ATTRIBUTE_MODEL_DIR, config.MODELFIT)

    # Database settings
    engine = config.ENGINE
    dbName = config.DB_NAME
    query = ''' SELECT * FROM %s.dbo.Product ''' % dbName
    # query = '''SELECT * FROM "%s".public."Product"''' % dbName

    productDF = pd.read_sql_query(query, engine)
    # Select only unlabeled products
    productDF = productDF.loc[productDF.loc[:,(config.ATTRIBUTE_COLUMNS)].fillna(value=0).astype('int64').sum(axis=1) != len(config.ATTRIBUTE_COLUMNS)]
    # Each entry
    logger.info("Executing product attribute annotation for %s unlabeled products" % len(productDF))
    for index, row in productDF.iterrows():
        if index==1:
            break
        # check if there is a blob or to skip it
        if row['Image'] is not None:
            image = helper.convertBlobToImage(row['Image'])
            image = helper.convertCVtoPIL(image)
            # Neckline
            if row['NeckDesign'] == None:
                productDF.loc[index, 'NeckDesign'] = helper.updateAttribute(config.DICTNECKLINE, image, necklineLearner)
            # Sleeve
            if row['Sleeve'] == None:
                productDF.loc[index, 'Sleeve'] = helper.updateAttribute(config.DICTSLEEVE, image, sleeveLearner)
            # Length
            if row['Length'] == None:
                productDF.loc[index, 'Length'] = helper.updateAttribute(config.DICTLENGTH, image, lengthLearner)
            # Collar
            if row['CollarDesign'] == None:
                productDF.loc[index, 'CollarDesign'] = helper.updateAttribute(config.DICTCOLLAR, image, collarLearner)
            # FIT
            if row['Fit'] == None:
                productDF.loc[index, 'Fit'] = helper.updateAttribute(config.DICTFIT, image, fitLearner)
            
    # Update Product table
    productDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
    # productDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(config.UPDATESQLQUERY)

    # End Counting Time
    logger.info("--- %s seconds ---" % (time.time() - start_time))