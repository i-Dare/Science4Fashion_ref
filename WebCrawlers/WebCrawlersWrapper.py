import os
import subprocess
import argparse
import sqlalchemy
import pandas as pd
import warnings

import helper_functions
import config

class WebCrawlers:

   def __init__(self):
      self.engine = helper_functions.ENGINE
      self.dbName = helper_functions.DB_NAME

      # Get all Adapters as stored in the database
      self.adapterDF = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter" % self.dbName, self.engine)
      # self.adapterDF = pd.read_sql_query("SELECT * FROM public.\"Adapter\"", self.engine)
      
      # Create a dictionary that maps each Adapter to the corresponding script
      # Get adapter scripts
      self.allAdapters = [adapter.lower() for adapter in self.adapterDF['Description'].values]
      # Get all python scripts in the WebCrawlers directory
      pythonScripts = [f for f in os.listdir(helper_functions.WEB_CRAWLERS) if str(f).endswith('.py')]
      adapterScripts = [f for f in pythonScripts if str(f).lower().rstrip('.py') in self.allAdapters]
      implementedAdapters = [a for a in self.allAdapters if a.lower() in 
            [f.lower().rstrip('.py') for f in adapterScripts]]
      # sort Adapters and scripts by name for grouping            
      implementedAdapters = sorted(implementedAdapters, key = lambda x: str(x).lower())
      adapterScripts = sorted(adapterScripts, key = lambda x: str(x).lower())
      # create Adapter dictionary
      self.adapter_dict = {a:os.path.join(helper_functions.WEB_CRAWLERS, f) for a,f in \
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
      self.parser.add_argument('--version', action = 'version', version = '%(prog)s  1.0')

      # Parse arguments
      self.args = self.parser.parse_args()
      self.adapter = self.args.adapter
      self.searchTerm = ' '.join(self.args.searchTerm)
      self.numberResults = self.args.numberResults

      # Setup argument constrains
      self.checkArgConstrains()


   # Check argument constrains      
   def checkArgConstrains(self,):
      for adapter in self.adapter:
         if adapter not in self.adapter_dict.keys():
            availableAdapters = [a for a in self.adapterDF['Description'].values if a.lower() in self.adapter_dict.keys()]
            self.parser.error('\nATTENTION: Adapter not implemented yet. Plese choose one of %s.' % availableAdapters)

      # Execute Website Crawler process
   def executeWebCrawler(self,):
      for adapter in self.adapter:
         print('Executing: %s' % self.adapter_dict[adapter])
         print('Search for %s on %s and return %s results' % (self.searchTerm, adapter, self.numberResults))
         # Execute Adapter      
         script = subprocess.Popen(['python', self.adapter_dict[adapter], self.searchTerm, str(self.numberResults)],
                                    stderr=subprocess.PIPE)
         _, err = script.communicate()                                    
         if err:
            warnings.warn(err.decode("utf-8"))

   # Execute text metadata based annotation  
   def executeTextBasedAnnotation(self,):
      print('Executing: text based annotation')
      scriptPath = os.path.join(helper_functions.TEXT_MINING, 'metadataAnnotation.py')
      script = subprocess.Popen(['python', scriptPath], stderr=subprocess.PIPE)
      _, err = script.communicate()                                    
      if err:
         warnings.warn(err.decode("utf-8"))

   # Execute color based annotation 
   def executeColorBasedAnnotation(self,):
      print('Executing: color based annotation')
      scriptPath = os.path.join(helper_functions.IMAGE_ANNOTATION, 'Color', 'colorAnnotation.py')
      script = subprocess.Popen(['python', scriptPath], stderr=subprocess.PIPE)
      _, err = script.communicate()                                    
      if err:
         warnings.warn(err.decode("utf-8"))

   # Execute clothing based annotation
   def executeClothingBasedAnnotation(self,):
      print('Executing: clothing based annotation')
      scriptPath = os.path.join(helper_functions.IMAGE_ANNOTATION, 'Clothing', 'clothingAnnotation.py')
      script = subprocess.Popen(['python', scriptPath], stderr=subprocess.PIPE)
      _, err = script.communicate()                                    
      if err:
         warnings.warn(err.decode("utf-8"))

   # Execute product clustering module
   def executeClustering(self, train=False):
      train_arg = '-train' if train else ''
      print('Executing: Clustering')
      scriptPath = os.path.join(helper_functions.CLUSTERING, 'clusteringConsensus.py')
      script = subprocess.Popen(['python', scriptPath, train_arg], stderr=subprocess.PIPE)
      _, err = script.communicate()                                    
      if err:
         warnings.warn(err.decode("utf-8"))

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
      