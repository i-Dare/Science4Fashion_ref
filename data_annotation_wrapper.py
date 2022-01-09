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
from AutoAnnotation.color import ColorAnnotation
from AutoAnnotation.clothing import ClothingAnnotation
from AutoAnnotation.text import MetadataAnnotation
from Clustering.clustering_consensus import ConsensusClustering

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
   
# ------------------------------------------------------------
#                     MODULE EXECUTION
# ------------------------------------------------------------
   # Execute product clustering module
   def executeClustering(self, train=False):
      self.logger.info('Executing: Clustering')      
      clustering = ConsensusClustering(
                                    user=self.user,
                                    linkage=config.LINKAGE,
                                    train=train,
                                    loglevel=self.loglevel)
      clustering.executeClustering()
      
      

   ## Sequencially executes the data annotation process
   # Step 1: Execute text metadata based annotation 
   # Step 2: Execute AutoAnnotation
   # Step 3: Execute product clustering module
   def run(self,):
      self.initAnnotation()
      executeAutoAnnotation(self.logger, self.user, self.oids, self.loglevel)
      # self.executeClustering(train=True)

# --------------------------------------------------------------------------                        
#          AutoAnnotation functionality
# --------------------------------------------------------------------------
# Execute AutoAnnotation module
def executeAutoAnnotation(logger, user, oids, loglevel):

   try:
      # Clothing annotation
      clothing_annotator = ClothingAnnotation.ClothingAnnotator(user, *oids, loglevel=loglevel) \
            if oids else ClothingAnnotation.ClothingAnnotator(user, loglevel=loglevel)
      clothing_annotator.execute_annotation()
   except Exception as e:
      logger.warn_and_trace(e)
      logger.warning('Clothing annotation for products failed')

   try:
      # Color annotation
      color_annotator = ColorAnnotation.ColorAnnotator(user, *oids, loglevel=loglevel) \
            if oids else ColorAnnotation.ColorAnnotator(user, loglevel=loglevel)
      color_annotator.execute_annotation()
   except Exception as e:
      logger.warn_and_trace(e)
      logger.warning('Color annotation for products not failed')

   try:
      # Metatada annotation
      metadata_annotator = MetadataAnnotation.MetadataAnnotator(user, *oids, loglevel=loglevel) \
            if oids else MetadataAnnotation.MetadataAnnotator(user, loglevel=loglevel)
      metadata_annotator.execute_annotation()
   except Exception as e:
      logger.warn_and_trace(e)
      logger.warning('Metadata annotation for products not failed')


if __name__ == "__main__":
   dataAnnotator = DataAnnotator()
   dataAnnotator.run()
      