import constants
import pandas as pd
import numpy as np
import sqlalchemy
import pymssql
import time

# It contains the code needed to convert product with id -> product with labels

def reconstructDF():
  start_time = time.time()
  ###Read Table Products from SocialMedia DB###
  engine = sqlalchemy.create_engine(constants.CONNECTION)
  query_db = pd.read_sql_query(constants.SELECTSQLQUERY, engine)
  assosDF = pd.DataFrame(query_db)
  ###Read all the tables of features from SocialMedia DB to match id to labels###
  for name in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
    locals()[str(name)+'_DB'] = pd.read_sql_query(constants.SELECTQUERY + str(name.upper()), engine)

  #Reconstruct assos to labelsDF to preprocess it in later steps
  labelsDF = pd.DataFrame(columns=['ProductNo'] + constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES)
  labelsDF['ProductNo'] = assosDF['ProductNo']
  locale = locals()
  print(labelsDF)
  print(assosDF)
  for col in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
    print(locale[str(col)+'_DB'])
    testing = assosDF.merge(locale[str(col)+'_DB'], how = 'left', on=(str(col)+'ID'))
    labelsDF.loc[:,str(col)] = testing.loc[:,str(col)]
  labelsDF.to_csv('/Users/Alexandros/Desktop/issel/scripts/databaseCustomization/ALLTESTING.csv')
  print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
  reconstructDF()