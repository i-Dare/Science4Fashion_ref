from operator import concat
import pandas as pd
import numpy as np
import time 
import pickle
import os
from datetime import datetime
import warnings
import argparse
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import entropy
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

import core.config as config
from core.helper_functions import *
from core.logger import S4F_Logger
from core.query_manager import QueryManager


class ConsensusClustering:

   def __init__(self, linkage='', train=False):
      #
      # Initialize argument parser
      #
      self.parser = argparse.ArgumentParser(description = 'A script for executing the Clustering \
            functionality', prog = 'ConsensusClustering')
      self.parser.add_argument('-train', help = '''Execute clustering by retraining the model, \
            if not selected, the clustering will be executed with the existing model''', 
            action="store_true", default = train)
      self.parser.add_argument('-l','--linkage', type = str, help = '''Input linkage type for the \
            Agglomerative clustering''', default = config.LINKAGE, choices=['ward', 'complete', 'average', 'sinlge'])
      self.parser.add_argument('-u', '--user', default=config.DEFAULT_USER, type = str, help = '''Input user''')

      # Parse arguments
      self.args, unknown = self.parser.parse_known_args()

      # Logger setup
      self.user = self.args.user
      self.logging = S4F_Logger('ClusteringLogger', user=self.user)
      self.logger = self.logging.logger
      self.helper = Helper(self.logging)

      # QueryManager setup
      self.db_manager = QueryManager(user=self.user)

      self.training_mode = self.args.train
      
      if self.training_mode:
         self.logger.info('Executing Consensus Clustering with training.')
      else:
         self.logger.info('Executing Consensus Clustering without training.')
      self.linkage = self.args.linkage      
      
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
      # Dictionary used for clustering evaluation
      self.clustering_dict = {}      


   def executeClustering(self):
      #
      ### Executes the training process for all the implemented clustering scenarios and exports 
      ### the final clustering model
      #
      # Begin Counting Time
      start_time = time.time()
      
      ## Preprocessing and attribute selection
      #   
      attributes_df = self.clustering_preprocessing()
      selected_data = self.attribute_selection(attributes_df)

      ## Factor Analysis of Mixed Data feature selection
      # 
      _, famd_featsDF = self.famd_features(selected_data)

      # Train consensus clustering model
      #
      labels, n_clusters, score = self.assignClusters(famd_featsDF)

      # Update clustering table
      self.update_clusters(labels, attributes_df)

      self.logger.info('''Number of final clusters: %s 
                           Final Silhouette Score: %s''' % (n_clusters, score))
      # End Counting Time
      self.logger.info("--- Finished clustering %s records in %s seconds ---" % (len(attributes_df), 
            round(time.time() - start_time, 2)))

   def clustering_preprocessing(self,):
      #
      ### Preprocessing
      #     
      # Collect information from the Product, ProductColor and ColorRGB tables of S4F database
      #
      self.logger.info('Start preprocessing of product attributes...')
      products_df = self._init_preprocessing()
      
      ## Merging results
      # 
      attributes_df = products_df.groupby('Oid').first()[config.PRODUCT_ATTRIBUTES + 
            ['Gender', 'RetailPrice', 'Description']]
      # Color data merging
      grouped_products_df = (products_df.groupby('Oid')
                .apply( lambda row:  list(row['Label'] ))
                .str.join(',')
                .str.split(',', expand=True))
      attributes_df.loc[:, ['ColorRanking%s' % n for n in grouped_products_df.columns]] = grouped_products_df.values

      for col in attributes_df:
         if attributes_df[col].nunique() == 1: # remove columns with identical elements
            del attributes_df[col]

         elif attributes_df[col].isnull().all():  # remove all-null columns
            del attributes_df[col]

         elif attributes_df[col].is_unique: # remove columns with unique elements
            del attributes_df[col]
      
      ## Fill NA and None values
      attributes_df[self.attributes].fillna(np.nan, inplace=True)
      return attributes_df

   def attribute_selection(self, data):
      # Select product attributes used in clustering, color ranking information and product description
      self.attributes += data.filter(like='ColorRanking').columns.tolist() + ['Description']
      attributes_df = data[set(self.attributes)]

      # Rearrange columns so that numerical attributes are last    
      num_columns = sorted(attributes_df._get_numeric_data().columns.tolist())
      cat_columns = sorted(list(set(attributes_df.columns) - set(num_columns) ))
      attributes_df = attributes_df[cat_columns + num_columns]
      # replace NaN values with NOT_APPLIED
      for col in cat_columns:                                          
         attributes_df[col].fillna('NOT_APPLIED', inplace=True)
      # replace numerical NaN values with the mean of the column
      for col in num_columns:                                          
         attributes_df[col].fillna(attributes_df[col].mean(), inplace=True)

      # Normalize numeric values
      attributes_df[num_columns] = MinMaxScaler().fit_transform(attributes_df[num_columns])
      return attributes_df

   def famd_features(self, data):
      self.logger.info('Start FAMD feature endcoding...')
      inertia = []
      for n in range(1, self.n_components):
         try:
            famd = prince.FAMD(check_input=True,n_iter = 3, n_components=n, random_state=42)
            famd.fit(data)
            # Calculate the 70th percentile for n number of components
            q = np.percentile(famd.explained_inertia_, 70, interpolation='nearest')
            where = np.where(famd.explained_inertia_ >= q)
            # Capture the number of components that explain the 70th percentile 
            inertia.append((len(famd.explained_inertia_[where]), 
                  sum(famd.explained_inertia_[where])/sum(famd.explained_inertia_)))       
        
         except:
            pass
      # Final number of FAMD components is the n for which the maximum explained inertia is captured 
      sorted_inertia = np.argsort([x[1] for x in inertia])[::-1]
      n_components, k_features = sorted_inertia[1] + 1, inertia[sorted_inertia[1]][0]
      self.logger.info('Executing optimized FAMD transformation for n=%s and selecting k=%s features' \
            % (n_components, k_features))
      # Execute optimized FAMD transformation
      famd = prince.FAMD(check_input=True, n_components=2, random_state=42)
      famd.fit(data)
      famd_featsDF = famd.row_coordinates(data)
      # Final feature selection
      famd_featsDF = famd_featsDF.loc[:, range(k_features)]

      return famd, famd_featsDF

   def assignClusters(self, data):
      #
      ### Executes the training process for all the implemented clustering scenarios and exports 
      ### the final clustering model
      #
      # Execute clustering
      #
      self.logger.info('Start clustering...')
      start_time = time.time() 
      self.run_clustering(data)
      #
      # Execute consensus clustering
      labels, n_clusters, score = self.consensus_clustering()
      self.logger.info('Finished clustering in %s seconds' % (round(time.time() - start_time, 2)))
      return labels, n_clusters, score
      
   def update_clusters(self, labels, data):
      ## Update Cluster table
      # Delete all existing records from Cluster table
      self.logger.info('Updating %s.Cluster table...' % config.DB_NAME)
      with self.engine.begin() as conn:
         conn.execute('''ALTER TABLE %s.dbo.Product NOCHECK CONSTRAINT FK_Product_Cluster''' % (self.dbName))
         conn.execute('''DELETE FROM %s.dbo.Cluster''' % (self.dbName))
         conn.execute('''ALTER TABLE  %s.dbo.Product CHECK CONSTRAINT FK_Product_Cluster''' % (self.dbName))

      # Insert new cluster labels    
      for label in set(labels):
         params = {'table': 'Cluster', 'Description': label}
         self.db_manager.runInsertQuery(params)

      ### Update Product table
      # Join cluster label and product ID information
      data['Clusters'] = labels
      # Reload Cluster table to get updated Cluster.Oid values
      query = ''' SELECT * FROM %s.dbo.Cluster ''' % self.dbName
      clusterDF = pd.read_sql_query(query, self.engine)
      clusterDF.loc[:, 'Description'] = clusterDF['Description'].apply(pd.to_numeric)
      # Add column to "data" with the Cluster Oid
      data['Cluster'] = data['Clusters'].map(clusterDF.set_index('Description')['Oid'].to_dict())
      start_time = time.time() 
      # Batch update Product table
      self.logger.info('Update Product Cluster')
      table = 'Product'
      columns = ['Oid', 'Cluster']
      self.db_manager.runBatchUpdate(table, data.reset_index()[columns], 'Oid')
      self.logger.info("Updated %s records in %s seconds" % (len(data), round(time.time() - start_time, 2)))

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
      attJoin, attSelect = '', []
      for i, attr in enumerate(config.PRODUCT_ATTRIBUTES):
         attSelect += ['attr%s.Description AS %s' % (i, attr)]
         attJoin += ' LEFT JOIN %s.dbo.%s AS attr%s\nON PRD.%s = attr%s.Oid ' \
               % (config.DB_NAME, attr, i, attr, i) 

      query = '''SELECT PRD.Oid, PRD.RetailPrice, PRD.Description,
                  C.Label, C.LabelDetailed, C.Red, C.Blue, C.Green,
                  PC.Ranking,
                  G.Description as Gender,
                  %s
            FROM %s.dbo.Product AS PRD 
               LEFT JOIN %s.dbo.ProductColor AS PC
               ON PC.Product=PRD.Oid
               LEFT JOIN %s.dbo.ColorRGB AS C
               ON PC.ColorRGB = C.Oid
               LEFT JOIN %s.dbo.Gender AS G
               ON PRD.Gender=G.Oid
               %s''' % (','.join(attSelect), config.DB_NAME, config.DB_NAME, config.DB_NAME, 
               config.DB_NAME, attJoin)  
      products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
      products_df.drop_duplicates(ignore_index=True, inplace=True)	
      return products_df

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
         self.helper.save_model(model_kmeans, config.MODEL_KMEANS, config.CLUSTERING_MODEL_DIR)
      else:
         model_kmeans = self.helper.get_model(config.MODEL_KMEANS)
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
         model_birch = self.helper.get_model(config.MODEL_BIRCH)
         model_birch.partial_fit(data)
         model_birch.predict(data)
      # save model
      self.helper.save_model(model_birch, config.MODEL_BIRCH, config.CLUSTERING_MODEL_DIR)
      self.clustering_dict[model_name] = model_birch.labels_

   def fuzzycmeans_clustering(self, data, init=2):
      model_name = 'Fuzzy C-Means'
      scaled_entropy_list = []

      for i in range(init, 31):
         _, labels, _, _, _, _, _ = skf.cmeans(data.T, i, 2, error=0.005, maxiter=1000, seed=42)
         scaled_entropy = entropy(labels, axis=1, base=2).sum()/np.log2(labels.shape[0])
         scaled_entropy_list.append(scaled_entropy)

      # Decide on the iptimal number of clusters according to Scaled Partition Entropy
      n_clusters = np.argmin(scaled_entropy_list) + init
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
         self.helper.save_model(model_dbscan, config.MODEL_DBSCAN, config.CLUSTERING_MODEL_DIR)
      else:
         model_dbscan = self.helper.get_model(config.MODEL_DBSCAN)
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
         self.helper.save_model(model_optics, config.MODEL_OPTICS, config.CLUSTERING_MODEL_DIR)         
      else:
         model_optics = self.helper.get_model(config.MODEL_OPTICS)
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
         self.helper.save_model(model_bgmm, config.MODEL_BGM, config.CLUSTERING_MODEL_DIR)
      else:
         model_bgmm = self.helper.get_model(config.MODEL_BGM)

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
      ##

      ## Calculate the similarity matrix
      sim_matrix_df = self._build_similarity_matrix()

      # Execute Agglomerative Clustering with ward linkage and the maximum of the minimum
      # 
      if self.training_mode:
         model_consensus = AgglomerativeClustering(n_clusters = None, 
                                         linkage=self.linkage,
                                         affinity='precomputed',
                                         distance_threshold=self.distance_threshold).fit(sim_matrix_df) 
         # Save model
         self.helper.save_model(model_consensus, config.MODEL_CONSENSUS, config.CLUSTERING_MODEL_DIR)
      else:
         model_consensus = self.helper.get_model(config.MODEL_CONSENSUS)
         model_consensus.fit_predict(sim_matrix_df)
      # Get labels
      labels = model_consensus.labels_
      n_clusters = model_consensus.n_clusters_
      score = metrics.silhouette_score(sim_matrix_df, labels)
      # Return clustering labels, numper of clusters and Silhouette score
      return labels, n_clusters, score

   # def fuzzy_clustering_approval()

   def clustering_approval(self, sils, davs, cals, model_name, init=2):
      scores_famd = np.hstack(np.dstack((sils, davs, cals)))
      norm_scores_famd = MinMaxScaler().fit_transform(scores_famd)

      ind_famd = np.argmax(norm_scores_famd[:,0] + (1 - norm_scores_famd[:,1]) + norm_scores_famd[:,2])
      
      self.logger.info('''
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
      # self.dbscan_clustering(data)
      # self.optics_clustering(data)
      # self.bayesian_gaussian_mixture_clustering(data)

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
      fig = px.scatter_3d(famd_data.sort_values('clustering'), x='axis_0', y='axis_1', z='axis_2',
                     color='clustering', title=title)
      fig.show()


if __name__ == "__main__":
   clustering = ConsensusClustering()
   clustering.executeClustering()
