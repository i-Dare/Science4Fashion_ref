import os
import subprocess
import argparse
import sqlalchemy
import pandas as pd
import warnings
from datetime import datetime
import time

from helper_functions import *
import config
# from logger import S4F_Logger

class WebCrawlers:
   def __init__(self):
      self.helper = Helper()
      self.engine = self.config.ENGINE
      self.dbName = self.config.DB_NAME
      self.logfile = 'tmp.log'

      # Get all Adapters as stored in the database
      self.adapterDF = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter" % self.dbName, self.engine)
      
      # Create a dictionary that maps each Adapter to the corresponding script
      # Get adapter scripts
      self.allAdapters = [adapter.lower() for adapter in self.adapterDF['Description'].values]
     
      # Get all python scripts in the WebCrawlers directory
      pythonScripts = [f for f in os.listdir(self.helper.WEB_CRAWLERS) if str(f).endswith('.py')]
      adapterScripts = [f for f in pythonScripts if str(f).lower().rstrip('.py') in self.allAdapters]
      implementedAdapters = [a for a in self.allAdapters if a.lower() in 
            [f.lower().rstrip('.py') for f in adapterScripts]]
      
      # sort Adapters and scripts by name for grouping            
      implementedAdapters = sorted(implementedAdapters, key = lambda x: str(x).lower())
      adapterScripts = sorted(adapterScripts, key = lambda x: str(x).lower())
      
      # create Adapter dictionary
      self.adapter_dict = {a:os.path.join(self.helper.WEB_CRAWLERS, f) for a,f in \
         zip(implementedAdapters, adapterScripts)}
      
      # Initialize argument parser      
      self.parser = argparse.ArgumentParser(description = 'A wrapper script for executing the website\
            crawlers', prog = 'Website Crawlers Wrapper')
      self.parser.add_argument('-s','--searchTerm', type = str, help = '''Input the search query term''', 
            required = True, nargs = '+')
      self.parser.add_argument('-n','--numberResults', type = int, help = '''Input the number of \
            results you want to return''', default = 10, nargs = '?')
      self.parser.add_argument('-a','--adapter', help = '''Input the Adapter you would like to use \
            for your search query. The Adapter should be one of: %s''' % self.adapterDF['Description'].values, 
            type = lambda s: s.lower(), choices = self.allAdapters, required = True, nargs = '+')  
      self.parser.add_argument('-u', '--user', help = '''User's name''') 
      self.parser.add_argument('--version', action = 'version', version = '%(prog)s  1.0')
      
      # Parse arguments
      self.args = self.parser.parse_args()
      self.adapter = self.args.adapter
      self.searchTerm = ' '.join(self.args.searchTerm)
      self.numberResults = self.args.numberResults
      self.user = self.args.user

      # Init logger
      self.logger = self.helper.initLogger('WrapperLogger', self.logfile)

      # Setup argument constrains
      self.checkArgConstrains()


   # Check argument constrains      
   def checkArgConstrains(self,):
      if not self.user:
         self.user = config.DEFAULT_USER
         self.logger.warning('Logging for default user')
      else:
         self.logger.warning('Logging for user: %s' % self.user)

      for adapter in self.adapter:
         if adapter not in self.adapter_dict.keys():
            availableAdapters = [a for a in self.adapterDF['Description'].values if a.lower() in self.adapter_dict.keys()]
            self.logger.warning('Adapter %s not implemented yet. Plese choose one of %s.' % (adapter, availableAdapters))


# ------------------------------------------------------------
#                     MODULE EXECUTION
# ------------------------------------------------------------
   # Execute Website Crawler process
   def executeWebCrawler(self,):
      for adapter in self.adapter:
         self.logger.info('Search for %s on %s' % (self.searchTerm, str(adapter).capitalize()))
         #                                                           
         # Execute Adapter    
         #       
         proc = subprocess.run(['python', 
                                 self.adapter_dict[adapter], 
                                 self.searchTerm, 
                                 str(self.numberResults),
                                 str(self.user),
                                 str(self.logfile)
                                 ],
                                 stderr=subprocess.STDOUT)
         if proc.returncode != 0:   
            self.logger.warning('Issues in adapter %s' % str(adapter).capitalize())


   # Execute text metadata based annotation  
   def executeTextBasedAnnotation(self,):
      self.logger.info('Executing: text based annotation')
      scriptPath = os.path.join(self.helper.TEXT_MINING, 'metadataAnnotation.py')      
      proc = subprocess.run(['python',
                              scriptPath, 
                              str(self.user),
                              ],
                              stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in text based annotation')
   

   # Execute color based annotation 
   def executeColorBasedAnnotation(self,):
      self.logger.info('Executing: color based annotation')
      scriptPath = os.path.join(self.helper.IMAGE_ANNOTATION, 'Color', 'colorAnnotation.py')
      proc = subprocess.run(['python',
                              scriptPath, 
                              str(self.user),
                              ],
                              stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in color based annotation')
      

   # Execute clothing based annotation
   def executeClothingBasedAnnotation(self,):
      self.logger.info('Executing: clothing based annotation')
      scriptPath = os.path.join(self.helper.IMAGE_ANNOTATION, 'Clothing', 'clothingAnnotation.py')
      proc = subprocess.run(['python',
                              scriptPath, 
                              str(self.user),
                              ],
                              stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in clothing based annotation')
      

   # Execute product clustering module
   def executeClustering(self, train=False):
      self.logger.info('Executing: Clustering')      
      scriptPath = os.path.join(self.helper.CLUSTERING, 'clusteringConsensus.py')
      if train:
         args = ['python', scriptPath, '-train', '--user', str(self.user)]
      else:
         args = ['python', scriptPath, '--user', str(self.user)]
      proc = subprocess.run(args, stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in clothing based annotation')
      

   ## Sequencially executes the data collection and annotation process
   # Step 1: Execute query for a selected website crawlers
   # Step 2: Execute text metadata based annotation 
   # Step 3: Execute color based annotation 
   # Step 4: Execute clothing based annotation
   # Step 5: Execute product clustering module
   def run(self,):
      self.executeWebCrawler()
      self.executeTextBasedAnnotation()
      self.executeColorBasedAnnotation()
      self.executeClothingBasedAnnotation()
      self.executeClustering(train=True)

if __name__ == "__main__":
   webCrawler = WebCrawlers()
   webCrawler.run()
      