import pyodbc
import pandas as pd
import numpy as np
import time 
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import sys

from sklearn import metrics
import sqlalchemy

import core.config as config
from core.helper_functions import *
from core.logger import S4F_Logger
from core.query_manager import QueryManager


def update_clusters(clustering_df, data):
    ## Update Cluster table
    # Delete all existing records from Cluster table
    logger.info('Updating %s.Cluster table...' % config.DB_NAME)

    with config.ENGINE.begin() as conn:
        conn.execute('''ALTER TABLE %s.dbo.Product NOCHECK CONSTRAINT FK_Product_Cluster''' % (config.DB_NAME))
        conn.execute('''DELETE FROM %s.dbo.Cluster''' % (config.DB_NAME))
        conn.execute('''ALTER TABLE  %s.dbo.Product CHECK CONSTRAINT FK_Product_Cluster''' % (config.DB_NAME))


    
    # Insert new cluster labels
    for i, row in clustering_df.iterrows():
        params = {
                    'table': 'Cluster', 
                    'Description': row['Cluster'],
                    'ProductCategory': row['ProductCategory'],
                    'ProductSubcategory': row['ProductSubcategory'],
                    'Length': row['Length'],
                    'Sleeve': row['Sleeve'],
                    'CollarDesign': row['CollarDesign'],
                    'NeckDesign': row['NeckDesign'],
                    'Fit': row['Fit'],
                    'Gender': row['Gender']
                }
        db_manager.runInsertQuery(params)

    ### Update Product table
    # Join cluster label and product ID information
    # Reload Cluster table to get updated Cluster.Oid values
    query = ''' SELECT * FROM %s.dbo.Cluster ''' % config.DB_NAME
    clusterDF = pd.read_sql_query(query, config.ENGINE)
    clusterDF.loc[:, 'Description'] = clusterDF['Description'].apply(pd.to_numeric)
    # Add column to "data" with the Cluster Oid
    data['Cluster'] = data['Clusters'].map(clusterDF.set_index('Description')['Oid'].to_dict())
    start_time = time.time() 
    # Batch update Product table
    logger.info('Update Product Cluster')
    table = 'Product'
    columns = ['Oid', 'Cluster']
    db_manager.runBatchUpdate(table, data.reset_index()[columns], 'Oid')
    logger.info("Updated %s records in %s seconds" % (len(data), round(time.time() - start_time, 2)))


if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time()
    user = sys.argv[1]
    # Logger setup
    logging = S4F_Logger('ClusteringCLLogger', user=user) 
    logger = logging.logger
    helper = Helper(logging)
    db_manager = QueryManager(user=user)
        
    attribute_list = config.PRODUCT_ATTRIBUTES + ['Gender']
    params = {'table': 'Product'}
    attributes_df = db_manager.runSelectQuery(params, filters=attribute_list + ['Oid'])
    attributes_df.set_index('Oid', inplace=True)
    attributes_df.fillna(-1, inplace=True)
    X = attributes_df.values

    _sils, _davs, _cals, cost = [], [], [], []
    init = 2
    model_name = 'KModes'
    tol = 10
    for k in range(init, 75):
        kmodes = KModes(n_clusters=k, init=config.INITKMODES, verbose=1, n_jobs=-1)
        clusters = kmodes.fit_predict(X)
        cost.append(kmodes.cost_)
        if len(cost)>2:
            if tol > np.abs(cost[-1] - cost[-2]):
                break

    # Collect and evaluate clustering
    n_clusters, score = np.argmin(cost) + init, min(cost)
    kmodes = KModes(n_clusters=n_clusters, init=config.INITKMODES, verbose=1, n_jobs=-1)
    clusters = kmodes.fit_predict(X)
    centroids = kmodes.cluster_centroids_
    logger.info('''Number of final clusters: %s Final Silhouette Score: %s''' % (n_clusters, score))

    # Update clustes
    clustering_dict = {attr:centroids[:,i] for i,attr in enumerate(attribute_list)}
    clustering_dict['Cluster'] = range(n_clusters)
    clustering_df = pd.DataFrame(clustering_dict)
    clustering_df.replace(-1, np.nan, inplace=True)

    attributes_df['Clusters'] = clusters

    update_clusters(clustering_df, attributes_df)

    logger.info("--- %s seconds ---" % (time.time() - start_time))
    