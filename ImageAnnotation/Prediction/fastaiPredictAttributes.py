from fastai.vision import *
import torch
import sqlalchemy
import pandas as pd
import time
import numpy as np

import helper_functions
import config
import warnings; warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning)

if __name__ == "__main__":
    #Begin Counting Time
    start_time = time.time()

    #Set Device
    defaults.device = torch.device(config.DEVICE)

    #Initialize Learners (5/5)
    necklineLearner = load_learner(config.ATTRIBUTE_MODELPATH, config.MODELNECKLINE)
    sleeveLearner = load_learner(config.ATTRIBUTE_MODELPATH, config.MODELSLEEVE)
    lengthLearner = load_learner(config.ATTRIBUTE_MODELPATH, config.MODELLENGTH)
    collarLearner = load_learner(config.ATTRIBUTE_MODELPATH, config.MODELCOLLAR)
    fitLearner = load_learner(config.ATTRIBUTE_MODELPATH, config.MODELFIT)

    #Each database
    engine = helper_functions.ENGINE
    dbName = helper_functions.DB_NAME
    # query = ''' SELECT * FROM %s.dbo.Product ''' % dbName
    query = '''SELECT * FROM "%s".public."Product"''' % dbName

    productDF = pd.read_sql_query(query, engine)

    productDF = productDF.loc[productDF.loc[:,(config.ATTRIBUTE_COLUMNS)].sum(axis=1) != len(config.ATTRIBUTE_COLUMNS)]
    #Each entry
    for index, row in productDF.iterrows():
        if index==1:
            break
        #check if there is a blob or to skip it
        if row['Image'] is not None:
            image = helper_functions.convertBlobToImage(row['Image'])
            image = helper_functions.convertCVtoPIL(image)
            #Neckline
            if row['NeckDesign'] == None:
                productDF.loc[index, 'NeckDesign'] = helper_functions.updateAttribute(config.DICTNECKLINE, image, necklineLearner)
            #Sleeve
            if row['Sleeve'] == None:
                productDF.loc[index, 'Sleeve'] = helper_functions.updateAttribute(config.DICTSLEEVE, image, sleeveLearner)
            #Length
            if row['Length'] == None:
                productDF.loc[index, 'Length'] = helper_functions.updateAttribute(config.DICTLENGTH, image, lengthLearner)
            #Collar
            if row['CollarDesign'] == None:
                productDF.loc[index, 'CollarDesign'] = helper_functions.updateAttribute(config.DICTCOLLAR, image, collarLearner)
            #FIT
            if row['Fit'] == None:
                productDF.loc[index, 'Fit'] = helper_functions.updateAttribute(config.DICTFIT, image, fitLearner)
            
            print(productDF.loc[index, config.ATTRIBUTE_COLUMNS])
    # Update Product table
    # productDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
    productDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(config.UPDATESQLQUERY)

    #End Counting Time
    print("--- %s seconds ---" % (time.time() - start_time))