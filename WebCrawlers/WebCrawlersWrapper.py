import os
import subprocess
import argparse
import sqlalchemy
import pandas as pd
import warnings
from datetime import datetime

from helper_functions import *
import config
# from logger import S4F_Logger

class WebCrawlers:

   def __init__(self):
      self.helper = Helper()

      self.engine = self.helper.ENGINE
      self.dbName = self.helper.DB_NAME

      # Get all Adapters as stored in the database
      self.adapterDF = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter" % self.dbName, self.engine)
      # self.adapterDF = pd.read_sql_query("SELECT * FROM public.\"Adapter\"", self.engine)
      
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
      self.initLogFile()

      # Setup argument constrains
      self.checkArgConstrains()

   def initLogFile(self,):
      now = datetime.now().strftime('%Y-%m-%d')
      self.logfile = '%s_%s.log' % (self.user, now)

   # Check argument constrains      
   def checkArgConstrains(self,):
      if not self.user:
         self.user = 'base'
         self.initLogFile()
         self.logger = self.helper.initLogger('BaseLogger', self.logfile)

         self.logger.info('Logging in base mode')

      for adapter in self.adapter:
         if adapter not in self.adapter_dict.keys():
            availableAdapters = [a for a in self.adapterDF['Description'].values if a.lower() in self.adapter_dict.keys()]
            self.logger.warning('Adapter %s not implemented yet. Plese choose one of %s.' % (adapter, availableAdapters))


# ------------------------------------------------------------
#                     MODULE EXECUTION
# ------------------------------------------------------------
   # Execute Website Crawler process
   def executeWebCrawler(self,):
      logger = self.helper.initLogger('WebCrawlerLogger', self.logfile)
      for adapter in self.adapter:
         logger.info('Search for %s on %s' % (self.searchTerm, str(adapter).capitalize()))
         #                                                           
         # Execute Adapter    
         #       
         script = subprocess.Popen(['python', 
                                       self.adapter_dict[adapter], 
                                       self.searchTerm, 
                                       str(self.numberResults),
                                       str(self.user),
                                       str(self.logfile)
                                    ],
                                    stderr=subprocess.PIPE)
         stdout, stderr = script.communicate()
         if script.poll()!=0:            
            logger.warning('WebCrawler for adapter %s failed' % str(adapter).capitalize())
            logger.warning(stderr.decode("utf-8"))
      logger.info('Subprocess finished')


   # Execute text metadata based annotation  
   def executeTextBasedAnnotation(self,):
      logger = self.helper.initLogger('TextAnnotationLogger', self.logfile)
      logger.info('Executing: text based annotation')
      scriptPath = os.path.join(self.helper.TEXT_MINING, 'metadataAnnotation.py')
      script = subprocess.Popen(['python', 
                                    scriptPath, 
                                    str(self.user),
                                    str(self.logfile)
                                    ],
                                    stderr=subprocess.PIPE)
      _, stderr = script.communicate()
      if script.poll()!=0:            
         logger.warning(stderr.decode("utf-8"))
      logger.info('Subprocess finished')


   # Execute color based annotation 
   def executeColorBasedAnnotation(self,):
      logger = self.helper.initLogger('ColorAnnotationLogger', self.logfile)
      logger.info('Executing: color based annotation')
      scriptPath = os.path.join(self.helper.IMAGE_ANNOTATION, 'Color', 'colorAnnotation.py')
      script = subprocess.Popen(['python', 
                                    scriptPath, 
                                    str(self.user),
                                    str(self.logfile)
                                    ],
                                    stderr=subprocess.PIPE)
      _, stderr = script.communicate()
      if script.poll()!=0:            
         logger.warning(stderr.decode("utf-8"))
      logger.info('Subprocess finished')


   # Execute clothing based annotation
   def executeClothingBasedAnnotation(self,):
      logger = self.helper.initLogger('ClothingAnnotationLogger', self.logfile)
      logger.info('Executing: clothing based annotation')
      scriptPath = os.path.join(self.helper.IMAGE_ANNOTATION, 'Clothing', 'clothingAnnotation.py')
      script = subprocess.Popen(['python', 
                                    scriptPath, 
                                    str(self.user),
                                    str(self.logfile)
                                    ],
                                    stderr=subprocess.PIPE)
      _, stderr = script.communicate()
      if script.poll()!=0:            
         logger.warning(stderr.decode("utf-8"))
      logger.info('Subprocess finished')


   # Execute product clustering module
   def executeClustering(self, train=False):
      logger = self.helper.initLogger('ClusteringLogger', self.logfile)
      logger.info('Executing: Clustering')
      train_arg = '-train' if train else ''
      scriptPath = os.path.join(self.helper.CLUSTERING, 'clusteringConsensus.py')
      script = subprocess.Popen(['python', 
                                    scriptPath,
                                    train_arg,
                                    str(self.user),
                                    str(self.logfile)
                                    ],
                                    stderr=subprocess.PIPE)
      _, stderr = script.communicate()
      if script.poll()!=0:            
         logger.warning(stderr.decode("utf-8"))
      logger.info('Subprocess finished')

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
      