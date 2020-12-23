import pandas as pd
import config
import os
import numpy as np

main_tables = ['Fit', 'CollarDesign', 'Length', 'Sleeve', 'Gender', 'NeckDesign', 'Adapter']

for table in main_tables:
   print('Populating table %s' % table)
   table_df =  pd.read_sql_query("SELECT * FROM %s.dbo.%s" % (config.DB_NAME, table), config.ENGINE)
   file_df = pd.read_csv(os.path.join(config.RESOURCESDIR, 'db_init', '%s.csv' % table))
   for value in file_df['Description'].unique():
      if value not in table_df['Description'].unique():
         print('Insert value %s in table %s' % (value, table))
         query = '''INSERT INTO %s.dbo.%s (Description)
                     VALUES ('%s');''' % (config.DB_NAME, table, value)
         with config.ENGINE.begin() as conn:
            conn.execute(query)
      