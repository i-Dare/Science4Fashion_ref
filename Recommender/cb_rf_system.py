import pandas as pd
from cb_rf import cb_rf
#from cb_rf import make_recommendation
from evaluator import evaluator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config
import os
import helper_functions

# read data
path = config.RECOMMENDER
engine = config.ENGINE
dbName = config.DB_NAME
dataset = pd.read_sql_query('''SELECT PRD.Oid, RSLT.Clicked, RSLT.IsFavorite, RSLT.GradeBySystem, RSLT.GradeByUser,RSLT.CreatedBy
                                FROM %s.dbo.PRODUCT AS PRD
                                LEFT JOIN %s.dbo.RESULT AS RSLT
                                ON PRD.oid = RSLT.Product''' % (dbName, dbName), engine)
dataset.rename(columns={'Oid': 'clothId'}, inplace=True)
dataset.insert(0,'userId',0)
columns =list(dataset.columns)
wholedata = dataset.copy()
dataset.drop(dataset[np.isnan(dataset.GradeByUser)].index, inplace=True)
print(dataset)
userID = 0;

# split train test

# initialize train dataset
all_train = pd.DataFrame(columns=columns)
# initialize train dataset
all_test = pd.DataFrame(columns=columns)
# get all unique users
all_users = dataset[columns[0]].unique()

# add 90% train 10% test from every user
tmp = dataset.loc[dataset[columns[0]] == userID]
train_X, test_X, train_y, test_y = train_test_split(tmp, tmp['GradeByUser'].tolist(), test_size=0.1, random_state=42)
all_train = all_train.append(train_X)
all_test = all_test.append(test_X)

# reset index for train and test set
all_test = all_test.reset_index().drop(columns='index')
all_train = all_train.reset_index().drop(columns='index')

dataset.head(5)
path2 = os.path.join(path, 'clothes_attr.pkl')
features =  pd.read_pickle(path2)
print(features)
rf = cb_rf(0,features, dataset, columns,wholedata)
predictedClothes = rf.make_recommendation()
print(predictedClothes)

# eval = evaluator()
# df = pd.DataFrame(columns=['user','metric','score'])
# df2 = pd.DataFrame(columns=['user','metric','score'])
# for u in range(1):
#   grade_df = rf.evaluate_system(all_train,all_test)
#   a,b = eval.average_arpf_rm(data=grade_df,cols=columns,threshold=5,model='rf')
#   df = df.append(a)
#   df2 = df2.append(b)