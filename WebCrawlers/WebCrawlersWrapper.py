import helper_functions
import config
import argparse
import sqlalchemy
import pandas as pd
import os
import subprocess

class WebCrawlers:

   def __init__(self):
      self.engine = helper_functions.ENGINE
      self.dbName = helper_functions.DB_NAME

      # Get all Adapters as stored in the database
      self.adapterDF = pd.read_sql_query("SELECT * FROM %s.dbo.Adapter" % self.dbName, self.engine)
      # self.adapterDF = pd.read_sql_query("SELECT * FROM public.\"Adapter\"", self.engine)
      
      # Get adapter scripts
      self.allAdapters = [adapter.lower() for adapter in self.adapterDF['description'].values]
      # Get all python scripts
      pythonScripts = [f for f in os.listdir(helper_functions.WEB_CRAWLERS) if str(f).endswith('.py')]
      adapterScripts = [f for f in pythonScripts if str(f).lower().rstrip('.py') in self.allAdapters]
      implementedAdapters = [a for a in self.allAdapters if a.lower() in 
            [f.lower().rstrip('.py') for f in adapterScripts]]
      # sort for grouping            
      implementedAdapters = sorted(implementedAdapters, key = lambda x: str(x).lower())
      adapterScripts = sorted(adapterScripts, key = lambda x: str(x).lower())
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
            for your search query. The Adapter should be one of: %s''' % self.adapterDF['description'].values, 
            type = lambda s: s.lower(), choices = self.allAdapters, required = True)   
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
      if self.adapter not in self.adapter_dict.keys():
         availableAdapters = [a for a in self.adapterDF['description'].values if a.lower() in self.adapter_dict.keys()]
         self.parser.error('\nATTENTION: Adapter not implemented yet. Plese choose one of %s.' % availableAdapters)


   # Executes the API queries
   def run(self,):
      print('Search for %s on %s and return %s results' % (self.searchTerm, 
                                                   self.adapter, 
                                                   self.numberResults))
      # Execute Adapter
      print('Executing: %s' % self.adapter_dict[self.adapter])
      subprocess.call(['python', self.adapter_dict[self.adapter], self.searchTerm, str(self.numberResults)])
      

if __name__ == "__main__":
   webCrawler = WebCrawlers()
   webCrawler.run()
      