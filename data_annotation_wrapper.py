import os
import subprocess
import argparse
import sqlalchemy
import pandas as pd
import warnings
from datetime import datetime
import time

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger
from core.query_manager import QueryManager

class DataAnnotator:
   def __init__(self):
      self.engine = config.ENGINE
      self.dbName = config.DB_NAME
      
      # Initialize argument parser      
      self.parser = argparse.ArgumentParser(description = 'A wrapper script for executing the adhoc \
            data annotation process', prog = 'Data Annotation Wrapper')
      self.parser.add_argument('-i','--id', help = '''Input the Product Oids for annotation. If \
            empty, the annotator will be executed for all Product Oids with empty product attributes''', 
            required = False, nargs = '+')  
      self.parser.add_argument('-u', '--user', required = True, help = '''User's name''')
      self.parser.add_argument('-l', '--loglevel', required = False, default=config.DEFAULT_LOGGING_LEVEL, help = '''Logging level''')
      
      # Parse arguments
      self.args = self.parser.parse_args()
      self.oids = self.args.id
      self.user = self.args.user      
      self.loglevel = self.args.loglevel      


   # Init function
   def initAnnotation(self,):
       # Init logger
      if not self.user:
         self.user = config.DEFAULT_USER        

      self.logging = S4F_Logger('AnnotationWrapperLogger', user=self.user, level=self.loglevel)
      self.logger = self.logging.logger
      # Init helper
      self.helper = Helper(self.logging)
      # Init db_manager
      self.db_manager = QueryManager(user=self.user)
   
   # Method for module execution
   def runProcess(self, scriptPath, process):
      self.logger.info('Executing: %s' % process)
      if self.oids:
         proc = subprocess.run(['python',
                              scriptPath, 
                              str(self.user),
                              *self.oids,
                              self.loglevel
                              ],
                              stderr=subprocess.STDOUT)
      else:
         proc = subprocess.run(['python',
                                 scriptPath, 
                                 str(self.user),
                                 self.loglevel
                                 ],
                                 stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in %s' % process)


# ------------------------------------------------------------
#                     MODULE EXECUTION
# ------------------------------------------------------------
     # Execute text metadata based annotation  
   def executeTextBasedAnnotation(self,):
      scriptPath = os.path.join(config.AUTOANNOTATION, 'text', 'MetadataAnnotation.py')
      self.runProcess(scriptPath, 'text based annotation')
   

   # Execute color based annotation 
   def executeColorBasedAnnotation(self,):
      scriptPath = os.path.join(config.AUTOANNOTATION, 'color', 'ColorAnnotation.py')
      self.runProcess(scriptPath, 'text based annotation')
      

   # Execute clothing based annotation
   def executeClothingBasedAnnotation(self,):
      scriptPath = os.path.join(config.AUTOANNOTATION, 'clothing', 'ClothingAnnotation.py')
      self.runProcess(scriptPath, 'text based annotation')
      

   # Execute product clustering module
   def executeClustering(self, train=False):
      self.logger.info('Executing: Clustering')      
      scriptPath = os.path.join(config.CLUSTERING, 'clustering_consensus.py')
      if train:
         args = ['python', scriptPath, '-train', '--user', str(self.user)]
      else:
         args = ['python', scriptPath, '--user', str(self.user)]
      proc = subprocess.run(args, stderr=subprocess.STDOUT)
      if proc.returncode != 0:
         self.logger.warning('Issues in clothing based annotation')
      

   ## Sequencially executes the data annotation process
   # Step 1: Execute text metadata based annotation 
   # Step 2: Execute color based annotation 
   # Step 3: Execute clothing based annotation
   # Step 4: Execute product clustering module
   def run(self,):
      self.initAnnotation()      
      self.executeColorBasedAnnotation()
      self.executeClothingBasedAnnotation()
      self.executeTextBasedAnnotation()
      self.executeClustering(train=True)


if __name__ == "__main__":
   dataAnnotator = DataAnnotator()
   dataAnnotator.run()
      