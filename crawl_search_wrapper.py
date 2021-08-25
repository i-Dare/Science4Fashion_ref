import os
import subprocess
import argparse
import pandas as pd
import sys, inspect, importlib
from datetime import datetime
from textblob import TextBlob

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager
from data_annotation_wrapper import executeAutoAnnotation


class WebCrawlers:
   def __init__(self):
      self.engine = config.ENGINE
      self.dbName = config.DB_NAME
      #
      # Get all Adapters as stored in the database
      self.adapterDF = QueryManager().runSelectQuery(params={'table': 'Adapter'})
      self.allAdapters = self.adapterDF['Description'].str.lower().tolist()     
      #
      # Initialize argument parser      
      self.parser = argparse.ArgumentParser(description = 'A wrapper script for executing the website\
            crawlers', prog = 'Website Crawlers Wrapper')
      self.parser.add_argument('-i','--id', type = int, help = '''Input the search id''', required = True)
      self.parser.add_argument('-l', '--loglevel', required = False, default=config.DEFAULT_LOGGING_LEVEL, help = '''Logging level''')
      #
      # Parse arguments
      self.args = self.parser.parse_args()
      self.crawlSearchID = self.args.id
      self.loglevel = self.args.loglevel      
      #
      # Initialize crawling
      self.initCrawling()


   # Init function
   def initCrawling(self,):
      # Fetch search information from CrawlSearch table
      search_df = QueryManager().runSelectQuery(params={'table': 'CrawlSearch', 'Oid': self.crawlSearchID})
      #
      # Assign query parameters to the parameters "searchTerm", "NumberOfProductsToReturn", "user", 
      # and "adapters"      
      self.numberResults = search_df.iloc[0]['NumberOfProductsToReturn']
      self.user = search_df.iloc[0]['UpdatedBy']
      self.adapters = [col.lower() for col in search_df.columns 
            if search_df.iloc[0][col] and col.lower() in self.allAdapters]
      self.initSearchTerm = search_df.iloc[0]['InitialSearchTerm']
      self.disableSpellCheck = search_df.iloc[0]['DisableSpellCheck']
      #
      # Init logger
      self.logging = S4F_Logger('WrapperLogger', user=self.user, level=self.loglevel)
      self.logger = self.logging.logger
      # Init helper
      self.helper = Helper(self.logging)
      # Init db_manager
      self.db_manager = QueryManager(user=self.user)
      #
      # Check spelling option
      self.spellCheck()
   
   def spellCheck(self,):
      if self.disableSpellCheck:
         self.searchTerm = self.initSearchTerm
      else:
         # Perform spelling correction
         self.searchTerm = TextBlob(self.initSearchTerm).correct().string
      # Update CrawlSearch record with search term
      uniq_params = {'table': 'CrawlSearch', 'Oid': self.crawlSearchID}
      params = {'table': 'CrawlSearch', 'SearchTerm': self.searchTerm}
      self.db_manager.runCriteriaUpdateQuery(uniq_params=uniq_params, params=params)

   # Update CrawlSearch table
   def updateCrawlSearchTable(self, description, adapter, numberResults):
      # Get adapter information from DB
      adapter_row = self.adapterDF[(self.adapterDF["Description"].str.lower()
                                        .str.fullmatch(str(adapter).lower()))]
      #                                                           
      # Update CrawlSearch table
      #  
      params = {'table': 'CrawlSearch', 
                'Description': description, 
                'Adapter': int(adapter_row.iloc[0]['Oid']),
                'NumberOfProductsToReturn': numberResults,
                'SearchTerm':description}
      self.db_manager.runInsertQuery(params)


# ------------------------------------------------------------
#                     MODULE EXECUTION
# ------------------------------------------------------------
   # Execute Website Crawler process
   def executeWebCrawler(self,):
      # Get all modules from SocialMediaCrawlers
      socialMediaCrawlersMods = [obj  for _, obj in inspect.getmembers(sys.modules['WebCrawlers.SocialMedia']) 
            if inspect.ismodule(obj)]
      # Get all modules from SocialMediaCrawlers
      websiteCrawlersMods = [obj  for _, obj in inspect.getmembers(sys.modules['WebCrawlers.Websites']) 
            if inspect.ismodule(obj)]
      adapterModules = socialMediaCrawlersMods + websiteCrawlersMods
      adapterModulesDict = {name: obj for mod in adapterModules 
            for name, obj in inspect.getmembers(sys.modules[mod.__name__])   
            if inspect.isclass(obj) and name.endswith('Crawler')}
      # Create a list with all the adapter classes
      adapterClassList = [
            list(filter(re.compile('%s' % a, re.I).match, list(adapterModulesDict.keys())))[0]  
            for a in self.adapters if list(filter(re.compile('%s' % a, re.I).match, 
                  list(adapterModulesDict.keys())))
         ]

      # Store in a list the Product IDs for all the newly added products to be later used during 
      # Auto Annotation
      self.oids = []
      for adapter in adapterClassList:
         self.logger.info('Execute search for "%s" on "%s"' % (self.searchTerm, adapter), 
               {'CrawlSearch': self.crawlSearchID})

         #                                                           
         # Programmatically execute adapter    
         #
         adapterClass = getattr(
               importlib.import_module(adapterModulesDict[adapter].__dict__['__module__']), adapter)(
                     self.crawlSearchID, 
                     self.searchTerm, 
                     self.numberResults, 
                     self.user, 
                     self.loglevel)
         productIDs = adapterClass.executeCrawling()
         self.oids += productIDs

   # Execute product clustering module
   def executeClustering(self, train=False):
      self.logger.info('Executing: Clustering', {'CrawlSearch': self.crawlSearchID})      
      scriptPath = os.path.join(config.CLUSTERING, 'clustering_consensus.py')
      if train:
         args = ['python', scriptPath, '-train', '--user', str(self.user), '--loglevel', self.loglevel]
      else:
         args = ['python', scriptPath, '--user', str(self.user), '--loglevel', self.loglevel]
      proc = subprocess.run(args, stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in clothing based annotation', {'CrawlSearch': self.crawlSearchID})
      

   ## Sequencially executes the data collection and annotation process
   # Step 1: Execute query for a selected website crawlers
   # Step 2: Execute AutoAnnotation
   # Step 3: Execute product clustering module
   def run(self,):
      self.executeWebCrawler()
      executeAutoAnnotation(self.logger, self.user, self.oids, self.loglevel)
      self.executeClustering(train=True)

if __name__ == "__main__":
   webCrawler = WebCrawlers()
   webCrawler.run()
      