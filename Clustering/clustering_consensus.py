from operator import concat
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime
import warnings
import argparse

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.cluster import KMeans, Birch, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn import mixture
import skfuzzy as skf

import seaborn as sns
import matplotlib.pyplot as plt

import prince 
from light_famd import FAMD

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger

class ConsensusClustering:

   def __init__(self, linkage='', train=False):
      #
      # Initialize argument parser
      #
      self.parser = argparse.ArgumentParser(description = 'A script for executing the Clustering \
            functionality', prog = 'ConsensusClustering')
      self.parser.add_argument('-train', help = '''Execute clustering by retraining the model, \
            if not selected, the clustering will be executed with the exesting model''', 
            action="store_true")
      self.parser.add_argument('-l','--linkage', type = str, help = '''Input linkage type for the \
            Agglomerative clustering''', default = config.LINKAGE, choices=['ward', 'complete', 'average', 'sinlge'])
      self.parser.add_argument('-u', '--user', type = str, help = '''Input user''')

      # Parse arguments
      self.args = self.parser.parse_args()
      self.training_mode = self.args.train
      self.linkage = self.args.linkage
      self.user = self.args.user
      
      ## Get initial configuration from "config" package
      #      
      # Get product clustering attributes
      self.attributes = config.CLUSTERING_PRODUCT_ATTRIBUTES
      # Get number of FAMD components
      self.n_components = config.FAMD_COMPONENTS

      # Get distance thereshold
      self.distance_threshold = config.DISTANCE_THRESHOLD
      # Database settings
      self.dbName = config.DB_NAME
      self.engine = config.ENGINE
      # Clustering model directory
      self.clustering_model_dir = config.CLUSTERING_MODEL_DIR
      # Dictionary used mapping products to IDs
      self.productOID_dict = {}
      # Dictionary used for clustering evaluation
      self.clustering_dict = {}

   def executeClustering(self):
      #
      ### Executes the training process for all the implemented clustering scenarios and exports 
      ### the final clustering model
      #
      # Begin Counting Time
      start_time = time.time()
      
      ## Preprocessing
      #   
      attributesDF = self.clustering_preprocessing()

      ## Factor Analysis of Mixed Data feature selection
      # 
      famd_featsDF = self.famd_features(attributesDF)

      # Train consensus clustering model
      #
      labels, n_clusters, score = self.assignClusters(famd_featsDF)

      # Update clustering table
      self.update_clusters(labels)

      logger.info('Number of final clusters: %s Final Silhouette Score: %s' % (n_clusters, score))
      # End Counting Time
      logger.info("--- %s seconds ---" % (time.time() - start_time))

   def clustering_preprocessing(self,):
      #
      ### Preprocessing
      #     
      # Collect information from the Product, ProductColor and ColorRGB tables of S4F database
      #
      logger.info('Start preprocessing of product attributes...')
      productDF, productColorDF, colorRGBDF = self._init_preprocessing()
      self.productOID_dict = productDF['Oid'].to_dict()

      # Merge color information from tables
      colorDF = productColorDF.merge(colorRGBDF, right_on='Oid', left_on='ColorRGB')      

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
      self.attributes += productDF.filter(like='ColorRanking').columns.tolist() + ['Description']
      attributesDF = productDF[set(self.attributes)]

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
         query = ''' SELECT * FROM %s.dbo.%s ''' % (self.dbName, attr)
         # query = '''SELECT * FROM "%s".public."%s"''' % (self.dbName, attr)
         attrDFDict[str(attr)+'DF'] = pd.read_sql_query(query, self.engine)

      # Merging product attribute tables to form the product table with the actual attributes and not the attribute ids.
      for attr in (feat_columns):
         if attr in attributesDF.columns: # chech in case column was removed during data cleaning
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

   def famd_features(self, data):
      logger.info('Start FAMD feature endcoding...')
      try:
         famd = prince.FAMD(check_input=True, n_components=self.n_components, random_state=42)
         famd.fit(data)
         famd_featsDF = famd.row_coordinates(data)
      except:
         famd_featsDF = self.famd_features(data)

      return famd_featsDF

   def assignClusters(self, data):
      #
      ### Executes the training process for all the implemented clustering scenarios and exports 
      ### the final clustering model
      #
      # Execute clustering
      #
      logger.info('Start clustering...')
      self.run_clustering(data)
      #
      # Execute consensus clustering
      labels, n_clusters, score = self.consensus_clustering()

      
      return labels, n_clusters, score

   def save_model(self, model, model_name):
      # Create model directory
      if not os.path.exists(self.clustering_model_dir):      
         os.makedirs(self.clustering_model_dir)
      model_path = os.path.join(self.clustering_model_dir, '%s.pkl' % model_name)
      pickle.dump(model, open(model_path, 'wb'))

   def get_model(self, model_name):
      try:
         model_list_paths = [os.path.join(config.CLUSTERING_MODEL_DIR, mname) 
                           for mname in os.listdir(config.CLUSTERING_MODEL_DIR) if model_name in mname]
         model_path = max(model_list_paths, key = os.path.getctime)
         model = pickle.load(open(str(model_path).replace('\\', os.sep), 'rb'))
         return model
      except:
         warnings.warn('Model %s not found in directory %s. Make sure the models are previously \
               trained, if the models have not been trained, execute class with argument \
               "train=True"' % (model_name, config.CLUSTERING_MODEL_DIR))
      
   def update_clusters(self, labels):
      ## Update Cluster table
      # Delete all existing records from Cluster table
      logger.info('Start updating S4F database...')
      with self.engine.begin() as conn:
         conn.execute('''ALTER TABLE %s.dbo.Product NOCHECK CONSTRAINT FK_Product_Cluster''' % (self.dbName))
         conn.execute('''DELETE FROM %s.dbo.Cluster''' % (self.dbName))
         conn.execute('''ALTER TABLE  %s.dbo.Product CHECK CONSTRAINT FK_Product_Cluster''' % (self.dbName))

         # conn.execute('''DELETE FROM "%s".public."Cluster" *''' % (self.dbName))
      # Insert new cluster labels    
      for label in set(labels):
         query = '''INSERT INTO %s.dbo.Cluster ("Description") VALUES (%s)''' % (self.dbName, label)
         # query = '''INSERT INTO "%s".public."Cluster" ("Description") VALUES (%s)''' % (self.dbName, label)
         with self.engine.begin() as conn:
            conn.execute(query)

      #  ## Update Product table
      # Join cluster label and product ID information
      clusterSeries = pd.Series({i:k for i,k in enumerate(labels)})
      productSeries = pd.Series(self.productOID_dict)
      productClusterDF = pd.concat([clusterSeries, productSeries], axis=1)
      # Reload Cluster table to get updated Cluster.Oid values
      query = ''' SELECT * FROM %s.dbo.Cluster ''' % self.dbName
      clusterDF = pd.read_sql_query(query, self.engine)
      clusterDF.loc[:, 'Description'] = clusterDF['Description'].apply(pd.to_numeric)
      # Add column to productClusterDF with the Cluster Oid
      productClusterDF['clusterID'] = productClusterDF[0].map(clusterDF.set_index('Description')['Oid'].to_dict())
      # Update Product table
      for _, row in productClusterDF.iterrows():
         label = row[0]
         oid = row[1]
         clusterID = row['clusterID']
         query = ''' UPDATE %s.dbo.Product SET Cluster=%s WHERE %s.dbo.Product.Oid=%s''' % (self.dbName, clusterID, self.dbName, oid)
         with self.engine.begin() as conn:
               conn.execute(query)

   def _build_similarity_matrix(self,):
      # Build the NxN consensus matrix from the clustering results
      _, consensus_matrix_df = self._build_consensus_matrix()

      # Create the Distance Matrix using the RBF kernel that measures the pairwise distance between items
      delta = .2
      sim_matrix_df = np.exp(-consensus_matrix_df ** 2 / (2. * delta ** 2))
      return sim_matrix_df

   def _build_consensus_matrix(self,):
      # Clustering DataFrame with the results of all the clustering algorithms
      clustering_df = pd.DataFrame.from_dict(self.clustering_dict)
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

   def _init_preprocessing(self,):
      #
      ### Loading product and color information from database
      selected_attr = ','.join(['\"%s\"' % s for s in self.attributes + ['Oid', 'Description']])
      query = ''' SELECT %s FROM %s.dbo.Product ''' % (selected_attr, self.dbName)
      # query = '''SELECT * FROM "%s".public."Product"''' % self.dbName
      productDF = pd.read_sql_query(query, self.engine)

      colorQuery = '''SELECT * FROM %s.dbo.ProductColor''' % self.dbName
      # colorQuery = '''SELECT * FROM public."ProductColor"''' 
      productColorDF = pd.read_sql_query(colorQuery, self.engine)

      colorRGBQuery = '''SELECT * FROM %s.dbo.ColorRGB''' % self.dbName
      # colorRGBQuery = '''SELECT * FROM public."ColorRGB"'''
      colorRGBDF = pd.read_sql_query(colorRGBQuery, self.engine)     

      return productDF, productColorDF, colorRGBDF

   # Clustering Algorithms executed with parameter tuning
   def kmeans_clustering(self, data, init=2):
      model_name = 'KMeans'
      _sils, _davs, _cals = [], [], []

      if self.training_mode:
         for i in range(init, 31):
            kmeans = KMeans(n_clusters=i, random_state=42, init='k-means++', n_jobs=-1).fit(data)
            
            _sils.append(metrics.silhouette_score(data, kmeans.labels_))
            _davs.append(metrics.davies_bouldin_score(data, kmeans.labels_))
            _cals.append(metrics.calinski_harabasz_score(data, kmeans.labels_))
         # Collect and evaluate clustering   
         _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
         n_clusters = self.clustering_approval(_sils, _davs, _cals, model_name)
         # Proposed clustering
         model_kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_jobs=-1).fit(data)
         # save model
         self.save_model(model_kmeans, config.MODEL_KMEANS)
      else:
         model_kmeans = self.get_model(config.MODEL_KMEANS)
         model_kmeans.predict(data)
      self.clustering_dict[model_name] = model_kmeans.labels_

   def birch_clustering(self, data, init=2):
      model_name = 'Birch'
      _sils, _davs, _cals = [], [], []

      if self.training_mode:
         for i in range(init, 31):
            birch = Birch(n_clusters=i).fit(data)
            
            _sils.append(metrics.silhouette_score(data, birch.labels_))
            _davs.append(metrics.davies_bouldin_score(data, birch.labels_))
            _cals.append(metrics.calinski_harabasz_score(data, birch.labels_))
         # Collect and evaluate clustering
         _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
         n_clusters = self.clustering_approval(_sils, _davs, _cals, model_name)
         # Proposed clustering
         model_birch = Birch(n_clusters=int(n_clusters)).fit(data)         
      else:
         model_birch = self.get_model(config.MODEL_BIRCH)
         model_birch.partial_fit(data)
         model_birch.predict(data)
      # save model
      self.save_model(model_birch, config.MODEL_BIRCH)
      self.clustering_dict[model_name] = model_birch.labels_

   def fuzzycmeans_clustering(self, data, init=2):
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
      n_clusters = self.clustering_approval(_sils, _davs, _cals, model_name)
      # Proposed clustering
      _, final_labels, _, _, _, _, _ = skf.cmeans(data.T, n_clusters, 2, error=0.005, maxiter=1000, seed=42)
      final_clusters = np.argmax(final_labels, axis=0)
      self.clustering_dict[model_name] = final_clusters

   def dbscan_clustering(self, data, init=2):
      model_name = 'DBSCAN'
      _sils, _davs, _cals = [], [], []

      if self.training_mode:
         for i in range(init, 31):
            dbscan = DBSCAN(eps=1, min_samples=i, n_jobs=-1).fit(data)

            _sils.append(metrics.silhouette_score(data, dbscan.labels_))
            _davs.append(metrics.davies_bouldin_score(data, dbscan.labels_))
            _cals.append(metrics.calinski_harabasz_score(data, dbscan.labels_))
         # Collect and evaluate clustering
         _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
         min_samples = self.clustering_approval(_sils, _davs, _cals, model_name)
         # Proposed clustering
         model_dbscan = DBSCAN(eps=1, min_samples=min_samples, n_jobs=-1).fit(data)
         # save model
         self.save_model(model_dbscan, config.MODEL_DBSCAN)
      else:
         model_dbscan = self.get_model(config.MODEL_DBSCAN)
         model_dbscan.fit_predict(data)
      self.clustering_dict[model_name] = model_dbscan.labels_

   def optics_clustering(self, data, init=2):
      model_name = 'OPTICS'
      _sils, _davs, _cals = [], [], []

      if self.training_mode:
         for i in range(init, 31):
            optics = OPTICS(min_samples=i, n_jobs=-1).fit(data)
            try:
               _sils.append(metrics.silhouette_score(data, optics.labels_))
               _davs.append(metrics.davies_bouldin_score(data, optics.labels_))
               _cals.append(metrics.calinski_harabasz_score(data, optics.labels_))
            except:
               warnings.warn('WARNING: for model %s, number of labels: %s' % 
                     (model_name, len(optics.labels_)))
         # Collect and evaluate clustering
         _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
         min_samples = self.clustering_approval(_sils, _davs, _cals, model_name)
         # Proposed clustering
         model_optics = OPTICS(min_samples=min_samples, n_jobs=-1).fit(data)
         # save model
         self.save_model(model_optics, config.MODEL_OPTICS)         
      else:
         model_optics = self.get_model(config.MODEL_OPTICS)
         model_optics.fit_predict(data)
      self.clustering_dict[model_name] = model_optics.labels_

   def bayesian_gaussian_mixture_clustering(self, data, init=2):
      model_name = 'Bayesian Gaussian Mixture'
      _sils, _davs, _cals = [], [], []

      if self.training_mode:
         for i in range(init, 31):
            bgmm = mixture.BayesianGaussianMixture(n_components=i, 
                                                   covariance_type='full', 
                                                   random_state=42).fit(data)
            labels = bgmm.predict(data)
            
            _sils.append(metrics.silhouette_score(data, labels))
            _davs.append(metrics.davies_bouldin_score(data, labels))
            _cals.append(metrics.calinski_harabasz_score(data, labels))
         # Collect and evaluate clustering
         _sils, _davs, _cals = np.asarray(_sils), np.asarray(_davs), np.asarray(_cals)
         n_components = self.clustering_approval(_sils, _davs, _cals, model_name)
         # Proposed clustering
         model_bgmm = mixture.BayesianGaussianMixture(n_components=n_components, 
                                                      covariance_type='full', 
                                                      random_state=42).fit(data)
         # save model
         self.save_model(model_bgmm, config.MODEL_BGM)
      else:
         model_bgmm = self.get_model(config.MODEL_BGM)

      final_labels = model_bgmm.predict(data)
      self.clustering_dict[model_name] = final_labels

   def consensus_clustering(self, ):
      ## Consensus (Aggregated) Clustering
      #
      #    Create a distance matrix based on the aggregated information of the different clustering 
      #    algorithms and use it as input to a Hierarchical Agglomerative Clustering model for the 
      #    final decision.
      #
      #    Implementation according to DOI: 10.1109/ICDM.2007.73
      #    ATTENTION: A final clustering solution may indicate structure where there is none, since we
      #              seek to force an ultimate clustering solution
      #
      ## Caclulate the similarity matrix
      sim_matrix_df = self._build_similarity_matrix()

      # Execute Agglomerative Clustering with ward linkage and the maximum of the minimum
      # 
      if self.training_mode:
         model_consensus = AgglomerativeClustering(n_clusters = None, 
                                         linkage=self.linkage,
                                         affinity='precomputed',
                                         distance_threshold=self.distance_threshold).fit(sim_matrix_df) 
         # Save model
         self.save_model(model_consensus, config.MODEL_CONSENSUS)
      else:
         model_consensus = self.get_model(config.MODEL_CONSENSUS)
         model_consensus.fit_predict(sim_matrix_df)
      # Get labels
      labels = model_consensus.labels_
      n_clusters = model_consensus.n_clusters_
      score = metrics.silhouette_score(sim_matrix_df, labels)
      # Return clustering labels, numper of clusters and Silhouette score
      return labels, n_clusters, score

   def clustering_approval(self, sils, davs, cals, model_name, init=2):
      scores_famd = np.hstack(np.dstack((sils, davs, cals)))
      norm_scores_famd = MinMaxScaler().fit_transform(scores_famd)

      ind_famd = np.argmax(norm_scores_famd[:,0] + (1 - norm_scores_famd[:,1]) + norm_scores_famd[:,2])
      
      logger.info('''
         Evaluation score of %s clustering:
         > Silhouette score: %s
         > Davies-Bouldin Index score: %s
         > Calinski-Harabasz Index score: %s
            ''' % (model_name, sils[ind_famd], davs[ind_famd], cals[ind_famd]))

      n_selection = ind_famd + init
      return n_selection
  
   def run_clustering(self, data):  
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
      self.kmeans_clustering(data)
      self.birch_clustering(data)
      self.fuzzycmeans_clustering(data)
      self.dbscan_clustering(data)
      self.optics_clustering(data)
      self.bayesian_gaussian_mixture_clustering(data)

   ## Evaluation functions
   def eval_affinity_matrix(self, matrix):
      fig, ax = plt.subplots(figsize = (11,7))
      sns.set_style('darkgrid')
      sns.distplot(matrix, bins=10)
      plt.show()

   def viz_clusters_3D(self, data, clustering, title=None):    
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


if __name__ == "__main__":
   # Logger setup
   logger = S4F_Logger('ClusteringLogger').logger
   helper = Helper(logger)

   clustering = ConsensusClustering()
   clustering.executeClustering()

