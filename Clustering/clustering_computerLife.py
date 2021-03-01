import pyodbc
import pandas as pd
import numpy as np
import time 
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn import metrics
import sqlalchemy

from core.helper_functions import *
import core.config as config

if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time()
    # Logger setup
    logger = S4F_Logger('ClusteringCLLogger').logger
    helper = Helper(logger)
    
    # Database settings
    engine = config.ENGINE
    dbName = config.DB_NAME
    query = ''' SELECT * FROM %s.dbo.Product ''' % dbName
    # query = '''SELECT * FROM "%s".public."Product"''' % dbName
    productDF = pd.read_sql_query(query, engine)
    
    labelsDF = productDF[config.CLUSTERING_PRODUCT_ATTRIBUTES].copy() #'Inspiration_Background'
    labelsDF.fillna(0, inplace=True)
    # Read all the tables of features from S4F DB to match id to genID
    attrDict = {}
    for attr in (config.PRODUCT_ATTRIBUTES):
        query = ''' SELECT * FROM %s.dbo.%s ''' % (dbName, attr)
        # query = '''SELECT * FROM "%s".public."%s"''' % (dbName, attr)
        attrDict[str(attr)+'DF'] = pd.read_sql_query(query, engine)
        
    # Merging tables with product to create the product with values not id.
    for attr in (config.PRODUCT_ATTRIBUTES):
        _tempDF = labelsDF.merge(attrDict[str(attr)+'DF'], how = 'left', left_on=str(attr), right_on='Oid')
        labelsDF.loc[:,str(attr)] = _tempDF.loc[:,str(attr)]
    
    ## Clustering process
    #
    # Prepare data for clustering (skip RetailPrice column)
    X = labelsDF.to_numpy()[:,1:].astype('int')
    N_CLUSTERS = 6
    kmodes = KModes(n_clusters=N_CLUSTERS, init=config.INITKMODES, verbose=2)
    clusters = kmodes.fit_predict(X)
    centroids = kmodes.cluster_centroids_
    clustering_dict = {attr:centroids[:,i] for i,attr in enumerate(labelsDF.columns[1:])}
    clustering_dict['Cluster'] = range(N_CLUSTERS)
    clustering_dict['RetailPrice'] = N_CLUSTERS * [None]
    clustering_dict['LifeStage'] = N_CLUSTERS * [None]
    clusteringDF = pd.DataFrame(clustering_dict)

    ## Update Cluster table
    #
    # clusteringDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
    clusteringDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(config.UPDATE_CLUSTERS_QUERY)

    ## Update Product table
    #
    dataDF = pd.DataFrame({'Oid': productDF['Oid'], 'Cluster': clusters})
    # dataDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
    dataDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(config.UPDATE_PRODUCT_CLUSTERS_QUERY) 
        
    # End Counting Time
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    
    

# Save to db:   a) a table with the product id and the cluster id: < df[['Product_No']], clusters >
#               b) a table with 6 rows (equal to the number of clusters) with the cendroids: < kmodes.cluster_centroids_ >




