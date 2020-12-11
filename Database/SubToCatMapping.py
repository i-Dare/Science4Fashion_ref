import sqlalchemy
import os
import json
import pandas as pd
import regex as re


CWD = os.getcwd()
PROJECT_HOME = os.environ['PROJECT_HOME']
PROJECT_CONFIG = os.path.join(PROJECT_HOME, 'config.json')
# Open project configuration file
with open(PROJECT_CONFIG) as f:
    config = json.load(f)
engine = sqlalchemy.create_engine(config['db_connection'] + config['db_name'])

ASK_SQL_Query1 = pd.read_sql_query("SELECT * FROM S4F.dbo.ProductCategory", engine)
cat = pd.DataFrame(ASK_SQL_Query1)
# cat = cat.loc[1:, :]

ASK_SQL_Query2 = pd.read_sql_query("SELECT * FROM S4F.dbo.ProductSubcategory", engine)
subcat = pd.DataFrame(ASK_SQL_Query2)
# subcat = subcat.loc[1:, :]

print('ss')
for index, sub in subcat.iterrows():
    for index2, c in cat.iterrows():
      # print(sub['Description'])
      # print(c['Description'])
      if(index !=0 and index2!=0):
        if (c["Description"] in sub["Description"]):
            sub["ProductCategory"] = c["Oid"]
            subcat.iloc[index] = sub
subcat.to_sql("temp_table", schema=config['db_name'] + '.dbo', con=engine, if_exists='replace', index=False)
sql = """UPDATE ProductSubcategory
            SET ProductSubcategory.ProductCategory = temp_table.ProductCategory
            FROM temp_table
            WHERE ProductSubcategory.Oid = temp_table.Oid"""

with engine.begin() as conn:
   conn.execute(sql)


print("sdfds")