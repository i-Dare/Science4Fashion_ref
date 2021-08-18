import os
import numpy as np
import pandas as pd

import core.config as config
from core.helper_functions import *


if __name__ == '__main__':

   main_db_tables = config.MAIN_DB_TABLES

   for table in main_db_tables:
      print('Populating table %s' % table)
      table_df =  pd.read_sql_query("SELECT * FROM %s.dbo.%s" % (config.DB_NAME, table), config.ENGINE)
      file_df = pd.read_csv(os.path.join(config.RESOURCESDIR, 'db_init', '%s.csv' % table))
      for value in file_df['Description'].unique():
         if value not in table_df['Description'].str.lower().unique():
            print('Insert value %s in table %s' % (value.lower(), table))
            query = '''INSERT INTO %s.dbo.%s (Description)
                        VALUES ('%s');''' % (config.DB_NAME, table, value.lower())
            with config.ENGINE.begin() as conn:
               conn.execute(query)
      