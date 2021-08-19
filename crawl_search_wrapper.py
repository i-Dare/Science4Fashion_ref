import os
import subprocess
import argparse
import pandas as pd
from datetime import datetime
from textblob import TextBlob

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager

class WebCrawlers:
   def __init__(self):
      self.engine = config.ENGINE
      self.dbName = config.DB_NAME
      
      # Get all Adapters as stored in the database
      self.adapterDF = QueryManager().runSelectQuery(params={'table': 'Adapter'})
      
      # Create a dictionary that maps each Adapter to the corresponding script
      # Get adapter scripts
      self.allAdapters = self.adapterDF['Description'].str.lower().tolist()
     
      # Get all python scripts in the WebCrawlers directory
      pythonScripts = [f for f in os.listdir(config.WEB_CRAWLERS) if str(f).endswith('.py')]
      adapterScripts = [f for f in pythonScripts if str(f).lower().rstrip('.py') in self.allAdapters]
      implementedAdapters = [a for a in self.allAdapters if a.lower() in 
            [f.lower().rstrip('.py') for f in adapterScripts]]
      
      # sort Adapters and scripts by name for grouping            
      implementedAdapters = sorted(implementedAdapters, key = lambda x: str(x).lower())
      adapterScripts = sorted(adapterScripts, key = lambda x: str(x).lower())

      # create Adapter dictionary
      self.adapter_dict = {a: os.path.join(config.WEB_CRAWLERS, f) for a,f in zip(implementedAdapters, 
            adapterScripts)}
      
      # Initialize argument parser      
      self.parser = argparse.ArgumentParser(description = 'A wrapper script for executing the website\
            crawlers', prog = 'Website Crawlers Wrapper')
      self.parser.add_argument('-i','--id', type = int, help = '''Input the search id''', required = True)
      self.parser.add_argument('-l', '--loglevel', required = False, default=config.DEFAULT_LOGGING_LEVEL, help = '''Logging level''')
      
      # Parse arguments
      self.args = self.parser.parse_args()
      self.crawlSearchID = self.args.id
      self.loglevel = self.args.loglevel      
      
      self.initCrawling()
      self.checkArgConstrains()


   # Init function
   def initCrawling(self,):
      # Fetch search information from CrawlSearch table
      search_df = QueryManager().runSelectQuery(params={'table': 'CrawlSearch', 'Oid': self.crawlSearchID})

      # Assign query parameters to the parameters "searchTerm", "NumberOfProductsToReturn", "user", 
      # and "adapters"      
      self.numberResults = search_df.iloc[0]['NumberOfProductsToReturn']
      self.user = search_df.iloc[0]['UpdatedBy']
      self.adapters = [col.lower() for col in search_df.columns if search_df.iloc[0][col] and col.lower() in self.allAdapters]
      self.initSearchTerm = search_df.iloc[0]['InitialSearchTerm']
      self.disableSpellCheck = search_df.iloc[0]['DisableSpellCheck']
      
      # Init logger
      self.logging = S4F_Logger('WrapperLogger', user=self.user, level=self.loglevel)
      self.logger = self.logging.logger
      # Init helper
      self.helper = Helper(self.logging)
      # Init db_manager
      self.db_manager = QueryManager(user=self.user)

      # Check spelling option
      self.spellCheck()

   # Check argument constrains      
   def checkArgConstrains(self,):
      for adapter in self.adapters:
         if adapter not in self.adapter_dict.keys():
            availableAdapters = [a for a in self.adapterDF['Description'].values if a.lower() in \
                  self.adapter_dict.keys()]
            self.logger.warning('Adapter %s not implemented yet. Plese choose one of %s.' \
                  % (adapter, availableAdapters), {'CrawlSearch': self.crawlSearchID})
   
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
      for adapter in self.adapters:
         self.logger.info('Execute search for %s on %s' % (self.searchTerm, str(adapter).capitalize()), 
               {'CrawlSearch': self.crawlSearchID})

         # Upadate CrawlSearch table
         # self.updateCrawlSearchTable(self.searchTerm, adapter, self.numberResults)

         #                                                           
         # Execute Adapter    
         #       
         proc = subprocess.run(['python', 
                                 self.adapter_dict[adapter], 
                                 str(self.crawlSearchID), 
                                 self.searchTerm, 
                                 str(self.numberResults),
                                 str(self.user),
                                 self.loglevel
                                 ],
                                 stderr=subprocess.STDOUT)
         if proc.returncode != 0:   
            self.logger.warning('Issues in adapter %s' % str(adapter).capitalize(), {'CrawlSearch': self.crawlSearchID}) 

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
   # Step 2: Execute product clustering module
   def run(self,):
      self.executeWebCrawler()
      # self.executeClustering(train=True)

if __name__ == "__main__":
   webCrawler = WebCrawlers()
   webCrawler.run()
      