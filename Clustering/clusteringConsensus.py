import pandas as pd
import numpy as np
import time 

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import KMeans, Birch, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn import mixture
import skfuzzy as skf


import seaborn as sns
import matplotlib.pyplot as plt

import sqlalchemy

import prince 
from light_famd import FAMD

import helper_functions
import config

def clustering_preprocessing(productDF, productColorDF, colorRGBDF, attributes, dbName, engine):
   #
   ### Preprocessing
   #
   ## Add color information on products   
   # Merge color info tables
   colorDF = productColorDF.merge(colorRGBDF, right_on='Oid', left_on='ColorRGP')

   # Merge color and product info tables
   mergeDF = colorDF.merge(productDF, right_on='Oid', left_on='Product')

   # Create color ranking columns in product tables
   for prno in mergeDF['Product'].unique():
      colors = []
      for _ in mergeDF['Ranking'].unique():
         colors = mergeDF.loc[mergeDF['Product']==prno]['LabelDetailed'].values
         for i,color in enumerate(colors, start=1):
               productDF.loc[productDF['Oid']==prno, 'ColorRanking%s' % i] = color

   ## Select desirable attributes of the merged table
   # Select product attributes used in clustering, color ranking information and product description
   attributes += productDF.filter(like='ColorRanking').columns.tolist() + ['Description']
   attributesDF = productDF[set(attributes)]

   for col in attributesDF:
      if attributesDF[col].nunique() == 1: # remove columns with identical elements
         del attributesDF[col]

      elif attributesDF[col].isnull().all():  # remove all-null columns
         del attributesDF[col]

      elif attributesDF[col].is_unique: # remove columns with unique elements
         del attributesDF[col]
   
   ## Fill NA and None values
   # Fill NA and NONE values
   attributesDF.fillna(np.nan, inplace=True)
   feat_columns = config.PRODUCT_ATTRIBUTES + ['Gender']

   # Read all the tables of features from S4F DB to match id to genID
   attrDFDict = {}
   for attr in (feat_columns):
      # query = ''' SELECT * FROM %s.dbo.%s ''' % (dbName, attr)
      query = '''SELECT * FROM "%s".public."%s"''' % (dbName, attr)
      attrDFDict[str(attr)+'DF'] = pd.read_sql_query(query, engine)

   # Merging attribut product tables to form the product table with the actual attributes and not the attribute ids.
   for attr in (feat_columns):
      attrDict = attrDFDict[str(attr)+'DF'][['Oid', 'Description']].set_index('Oid').to_dict()['Description']
      attributesDF.loc[attributesDF.loc[:, attr].isnull(), attr] = np.nan
      attributesDF.loc[:, attr] = attributesDF.loc[:, attr].map(int, na_action='ignore').map(attrDict, na_action='ignore').values.tolist()
      
   # rearrange columns so that numerical attributes are last    
   num_columns = sorted(attributesDF._get_numeric_data().columns.tolist())
   cat_columns = sorted(list(set(attributesDF.columns) - set(num_columns)))
   attributesDF = attributesDF[cat_columns + num_columns]
   # replace NaN values with NOT_APPLIED
   for col in cat_columns:                                          
      attributesDF[col].fillna('NOT_APPLIED', inplace=True)
   # replace numerical NaN values with the mean of the column
   for col in num_columns:                                          
      attributesDF[col].fillna(attributesDF[col].median(), inplace=True)

   # Normalize numeric values
   attributesDF[num_columns] = MinMaxScaler().fit_transform(attributesDF[num_columns])
   return attributesDF

def famd_features(data, n_components):
   try:
      famd = prince.FAMD(check_input=True, n_components=n_components, random_state=42)
      famd.fit(data)
      famd_featsDF = famd.row_coordinates(data)
   except:
      famd_featsDF = famd_features(data, n_components)

   return famd_featsDF

def kmeans_clustering(data, init=2):
   model_name = 'KMeans'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      kmeans = KMeans(n_clusters=i, random_state=42, init='k-means++').fit(data)
      
      _sils.append(metrics.silhouette_score(data, kmeans.labels_))
      _davs.append(metrics.davies_bouldin_score(data, kmeans.labels_))
      _cals.append(metrics.calinski_harabasz_score(data, kmeans.labels_))
   # Collect and evaluate clustering   
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   n_clusters = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   final_kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++').fit(data)
   clustering_dict[model_name] = final_kmeans.labels_

def birch_clustering(data, init=2):
   model_name = 'Birch'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      birch = Birch(n_clusters=i).fit(data)
      
      _sils.append(metrics.silhouette_score(data, birch.labels_))
      _davs.append(metrics.davies_bouldin_score(data, birch.labels_))
      _cals.append(metrics.calinski_harabasz_score(data, birch.labels_))
   # Collect and evaluate clustering
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   n_clusters = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   final_birch = Birch(n_clusters=int(n_clusters)).fit(data)
   clustering_dict[model_name] = final_birch.labels_

def fuzzycmeans_clustering(data, init=2):
   model_name = 'Fuzzy C-Means'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      _, labels, _, _, _, _, _ = skf.cmeans(data.T, i, 2, error=0.005, maxiter=1000, seed=42)
      clusters = np.argmax(labels, axis=0)
      
      _sils.append(metrics.silhouette_score(data, clusters))
      _davs.append(metrics.davies_bouldin_score(data, clusters))
      _cals.append(metrics.calinski_harabasz_score(data, clusters))
   # Collect and evaluate clustering
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   n_clusters = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   _, final_labels, _, _, _, _, _ = skf.cmeans(data.T, n_clusters, 2, error=0.005, maxiter=1000, seed=42)
   final_clusters = np.argmax(final_labels, axis=0)
   clustering_dict[model_name] = final_clusters

def dbscan_clustering(data, init=2):
   model_name = 'DBSCAN'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      dbscan = DBSCAN(eps=1, min_samples=i).fit(data)

      _sils.append(metrics.silhouette_score(data, dbscan.labels_))
      _davs.append(metrics.davies_bouldin_score(data, dbscan.labels_))
      _cals.append(metrics.calinski_harabasz_score(data, dbscan.labels_))
   # Collect and evaluate clustering
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   min_samples = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   final_dbscan = DBSCAN(eps=1, min_samples=min_samples).fit(data)
   clustering_dict[model_name] = final_dbscan.labels_

def optics_clustering(data, init=2):
   model_name = 'OPTICS'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      optics = OPTICS(min_samples=i).fit(data)
      
      _sils.append(metrics.silhouette_score(data, optics.labels_))
      _davs.append(metrics.davies_bouldin_score(data, optics.labels_))
      _cals.append(metrics.calinski_harabasz_score(data, optics.labels_))
   # Collect and evaluate clustering
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   min_samples = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   final_optics = OPTICS(min_samples=min_samples).fit(data)
   clustering_dict[model_name] = final_optics.labels_

def bayesian_gaussian_mixture_clustering(data, init=2):
   model_name = 'Bayesian Gaussian Mixture'
   _sils, _davs, _cals = [], [], []

   for i in range(init, 31):
      bgmm = mixture.BayesianGaussianMixture(n_components=i, covariance_type='full', random_state=42).fit(data)
      labels = bgmm.predict(data)
      
      _sils.append(metrics.silhouette_score(data, labels))
      _davs.append(metrics.davies_bouldin_score(data, labels))
      _cals.append(metrics.calinski_harabasz_score(data, labels))
   # Collect and evaluate clustering
   _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
   n_components = clustering_approval(_sils, _davs, _cals, model_name)
   # Proposed clustering
   final_bgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(data)
   final_labels = final_bgmm.predict(data)
   clustering_dict[model_name] = final_labels

def consensus_clustering(sim_matrix_df, linkage='average', distance_threshold=.5):
   # Agglomerative clustering for final consensus
   clustering = AgglomerativeClustering(n_clusters = None, 
                                        linkage=linkage,
                                        affinity='precomputed',
                                        distance_threshold=distance_threshold).fit(sim_matrix_df) 
   labels = clustering.labels_
   n_clusters = clustering.n_clusters_
   score = metrics.silhouette_score(sim_matrix_df, labels)
   # Return clustering labels, numper of clusters and Silhouette score
   return labels, n_clusters, score

def clustering_approval(sils, davs, cals, model_name, init=2):
   scores_famd = np.hstack(np.dstack((sils, davs, cals)))
   norm_scores_famd = MinMaxScaler().fit_transform(scores_famd)

   ind_famd = np.argmax(norm_scores_famd[:,0] + (1 - norm_scores_famd[:,1]) + norm_scores_famd[:,2])
   
   print('''
      Evaluation score of %s clustering:
      > Silhouette score: %s
      > Davies-Bouldin Index score: %s
      > Calinski-Harabasz Index score: %s
         ''' % (model_name, sils[ind_famd], davs[ind_famd], cals[ind_famd]))

   n_selection = ind_famd + init
   return n_selection

def build_consensus_matrix():
   # Clustering DataFrame with the results of all the clustering algorithms
   clustering_df = pd.DataFrame.from_dict(clustering_dict)
   # Start the clustering on the same basis from label 0 to N
   for col in clustering_df.columns:
      cluster_map = {c:i for i,c in zip(range(len(clustering_df[col].unique())), clustering_df[col].unique())}
      clustering_df.loc[:, col] = clustering_df[col].map(cluster_map)

   # Construct NxN Consensus Matrix
   consensus_matrix = np.zeros((len(clustering_df), len(clustering_df)))
   consensus_matrix_df = pd.DataFrame(consensus_matrix, 
                                      columns=range(len(consensus_matrix)), 
                                      index=range(len(consensus_matrix)))

   # iterate all clustering algorithms
   for algorithm in clustering_df.columns:
      # iterate all unique clusters
      for c in clustering_df[algorithm].unique():
         # iterate all items that belong to this cluster and increase by 1 the occurance index
         for i in clustering_df.loc[clustering_df[algorithm]==c].index:
               consensus_matrix_df.loc[i][clustering_df.loc[clustering_df[algorithm]==c].index] += 1

   # Get the co-occurance by dividing with the number of agorithms
   consensus_matrix_df = round(consensus_matrix_df/len(clustering_df.columns), 3)
   return clustering_df, consensus_matrix_df

## Evaluation functions
def eval_affinity_matrix(matrix):
   fig, ax = plt.subplots(figsize = (11,7))
   sns.set_style('darkgrid')
   sns.distplot(matrix, bins=10)
   plt.show()

def viz_clusters_3D(data, clustering, title=None):    
    import plotly.express as px 
    try:
        famd = prince.FAMD(check_input=True, n_components=3, random_state=42).fit(data)
    except:
        famd = prince.FAMD(check_input=True, n_components=3, random_state=42).fit(data)
    famd_data = famd.row_coordinates(data)
    famd_data.columns = ['axis_%s' %col for col in famd_data.columns]
    famd_data['clustering'] = clustering
    famd_data['clustering'] = clustering.astype(str)    
    fig = px.scatter_3d(famd_data, x='axis_0', y='axis_1', z='axis_2',
                  color='clustering', title=title)
    fig.show()

clustering_dict = {}

if __name__ == "__main__":
   # Begin Counting Time
   start_time = time.time()

   #Connect to database with sqlalchemy
   engine = helper_functions.ENGINE
   dbName = helper_functions.DB_NAME

   # Database settings
   engine = helper_functions.ENGINE
   dbName = helper_functions.DB_NAME

   #
   ### Loading product and color information from database
   # query = ''' SELECT * FROM %s.dbo.Product ''' % dbName
   query = '''SELECT * FROM "%s".public."Product"''' % dbName
   productDF = pd.read_sql_query(query, engine)

   # colorQuery = '''SELECT * FROM %s.dbo.ProductColor''' % dbName
   colorQuery = '''SELECT * FROM public."ProductColor"''' 
   productColorDF = pd.read_sql_query(colorQuery, engine)

   # colorRGBQuery = '''SELECT * FROM %s.dbo.ColorRGB''' % dbName
   colorRGBQuery = '''SELECT * FROM public."ColorRGB"'''
   colorRGBDF = pd.read_sql_query(colorRGBQuery, engine)
   
   ## Preprocessing
   #   
   attributesDF = clustering_preprocessing(productDF.copy(), 
                                           productColorDF.copy(), 
                                           colorRGBDF.copy(), 
                                           config.CLUSTERING_PRODUCT_ATTRIBUTES, 
                                           dbName, 
                                           engine)

   ## Factor Analysis of Mixed Data feature selection
   # 
   n_components = config.FAMD_COMPONENTS
   famd_featsDF = famd_features(attributesDF, n_components)

   #
   ### Clustering Algorithms
   #
   # Execute the 6 clustering models:
   #    * K-Means
   #    * Birch
   #    * Fuzzy C-Means
   #    * DBSCAN
   #    * OPTICS
   #    * Bayesian Gaussina Mixture Models
   kmeans_clustering(famd_featsDF, init=2)
   birch_clustering(famd_featsDF, init=2)
   fuzzycmeans_clustering(famd_featsDF, init=2)
   dbscan_clustering(famd_featsDF, init=2)
   optics_clustering(famd_featsDF, init=2)
   bayesian_gaussian_mixture_clustering(famd_featsDF, init=2)
   
   ## Consensus (Aggregated) Clustering
   #
   #    Create a distance matrix based on the aggregated information of the different clustering 
   #    algorithms and use it as input to a Hierarchical Agglomerative Clustering model for the 
   #    final decision.
   #
   #    Implementation according to DOI: 10.1109/ICDM.2007.73
   #    ATTENTION: A final clustering solution may indicate structure where is none, since we seek
   #               the ultimate clustering
   #
   # Build the NxN consensus matrix from the clustering results
   _, consensus_matrix_df = build_consensus_matrix()

   # Create the Distance Matrix using the RBF kernel that measures the pairwise distance between items
   delta = .2
   sim_matrix_df = np.exp(- consensus_matrix_df ** 2 / (2. * delta ** 2))

   # Execute Agglomerative Clustering with ward linkage and the maximum of the minimum
   linkage = config.LINKAGE
   distance_threshold = config.DISTANCE_THRESHOLD
   labels, n_clusters, score = consensus_clustering(sim_matrix_df, linkage, distance_threshold)
   print('Number of final clusters: %s \nFinal Silhouette Score: %s' % (n_clusters, score))

   ## Update Cluster table
   # Delete all existing records from ConsensusCluster table
   with engine.begin() as conn:
      # conn.execute('''DELETE FROM %s.dbo.ConsensusCluster *''' % (dbName))
      conn.execute('''DELETE FROM "%s".public."ConsensusCluster" *''' % (dbName))
   # Insert new cluster labels    
   for label in set(labels):
      # query = '''INSERT INTO %s.dbo.ConsensusCluster ("Description") VALUES (%s)''' % (dbName, label)
      query = '''INSERT INTO "%s".public."ConsensusCluster" ("Description") VALUES (%s)''' % (dbName, label)
      with engine.begin() as conn:
         conn.execute(query)

   #  ## Update Product table
   clusterSeries = pd.Series({i:k for i,k in enumerate(labels)})
   # Reload Product table
   # query = ''' SELECT * FROM %s.dbo.Product ''' % dbName
   query = '''SELECT * FROM "%s".public."Product"''' % dbName
   productDF = pd.read_sql_query(query, engine)
   # Update Product table
   productDF.loc[:, 'ConsensusCluster'] = clusterSeries
   # productDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
   productDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
   with engine.begin() as conn:
      conn.execute(config.UPDATE_PRODUCT_CONSENSUS_CLUSTERS_QUERY)
   
   # End Counting Time
   print("--- %s seconds ---" % (time.time() - start_time))
    



   




