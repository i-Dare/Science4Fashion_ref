from matplotlib.pyplot import figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.spatial
import numpy as np
import urllib.request as urllib2
from sklearn.feature_extraction.text import CountVectorizer
from skimage import io
import difflib
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import copy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from scipy.stats import pearsonr
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
import helper_functions

def novelty(user_id, items):
    products = data.loc[data['UserId'] == user_id]
    products = products.set_index('ProductNo')
    rec_items = products.loc[items]['clicks'].tolist()
    return round((len(rec_items) - len(np.array(np.nonzero(rec_items)).squeeze())) / len(rec_items), 2)


def coverage(total_rec, already_rated):
    return round((len(total_rec) / (4325 - already_rated)), 3)


def diversity(items):
    matrix = cosine_similarity(np.array(cloth_attr.loc[items]))
    return round(1 - np.mean(matrix[np.nonzero(np.triu(matrix, 1))]), 3)


import re


def lower_dimension_frame2(dataset, attr):
    # initialize category array
    categories = []
    for att in attr:
        # get all unique categories, split them from comma
        for i in dataset[att].unique():
            if type(i) == float:
                i = 'NAN'
            splited_ = re.split('/|,|to|and| |&|-|', i)
            splited_ = [x.lower() for x in splited_]
            for j in range(len(splited_)):
                if splited_[j] not in categories:
                    categories.append(splited_[j])

    # create a new array that contains all those categories
    result_frame = pd.DataFrame()
    result_array = np.zeros((len(dataset[attr]), len(categories)))
    for att in attr:
        for i in range(len(dataset[att])):
            if type(dataset[att].tolist()[i]) == float:
                splited_ = 'NAN'.split('/')
            else:
                splited_ = re.split('/|,|to|and| |&|-|', dataset[att].tolist()[i])
                splited_ = [x.lower() for x in splited_]
            for j in range(len(splited_)):
                for k in range(len(categories)):
                    if splited_[j] == categories[k]:
                        result_array[i][k] = 1

    result_frame = pd.DataFrame(result_array,
                                columns=categories)

    index_frame = dataset['ProductNo'].tolist()
    result_frame['index'] = index_frame
    result_frame = result_frame.set_index('index')
    return result_frame


def lower_dimension_frame(dataset, attr):
    # initialize category array
    categories = []
    for att in attr:
        # get all unique categories, split them from comma
        for i in dataset[att].unique():
            if type(i) == float:
                i = 'NAN'
            splited_ = i.split(',')
            for j in range(len(splited_)):
                if splited_[j] not in categories:
                    categories.append(splited_[j])

    # create a new array that contains all those categories
    result_frame = pd.DataFrame()
    result_array = np.zeros((len(dataset[attr]), len(categories)))
    for att in attr:
        for i in range(len(dataset[att])):
            if type(dataset[att].tolist()[i]) == float:
                splited_ = 'NAN'.split(',')
            else:
                splited_ = dataset[att].tolist()[i].split(',')
            for j in range(len(splited_)):
                for k in range(len(categories)):
                    if splited_[j] == categories[k]:
                        result_array[i][k] = 1

    result_frame = pd.DataFrame(result_array,
                                columns=categories)

    index_frame = dataset['ProductNo'].tolist()
    result_frame['index'] = index_frame
    result_frame = result_frame.set_index('index')
    return result_frame


def user_input_dataset():
    # load dataset to temporary variable
    temp_dataset = data

    # available search words in dataset
    all_searchWords = data['SearchWords'].tolist()

    # make search words to lower case
    all_searchWords = [x.lower() for x in all_searchWords]

    # remove duplicate search words
    all_searchWords = set(all_searchWords)

    # info about search words
    print('There are total: ', len(all_searchWords), ' search words ')

    # user searching clothes category
    input_searchWords = input("Please enter your search words:  ")

    # user searching clothes category
    numOf_relatedWords = input("Please enter the number of releated words we wanna search :  ")
    numOf_relatedWords = int(numOf_relatedWords)

    # find matches based on dataset searchwords
    matches = difflib.get_close_matches(input_searchWords, all_searchWords, numOf_relatedWords)

    # print matches
    print('Result found: ', matches)

    # merge result of matches
    dataFrames = []
    for i in matches:
        dataFrames.append(data.loc[data['SearchWords'] == i.upper()])

    # all available clothes based on user search
    totalClothes = pd.concat(dataFrames)

    temp_dataset = totalClothes

    return temp_dataset


def frame_merger(dataset):
    merged_frame = pd.merge(prod_frame, fit_frame, on='index')
    # merged_frame = pd.merge(merged_frame,length_frame,on='index')
    # merged_frame = pd.merge(merged_frame,color_frame,on='index')
    merged_frame = pd.merge(merged_frame, neck_frame, on='index')
    merged_frame = pd.merge(merged_frame, sleeve_frame, on='index')
    merged_frame = pd.merge(merged_frame, search_word_Dataframe, on='index')
    merged_frame['url'] = dataset['ImageSource'].tolist()
    merged_frame['gradeSystem'] = dataset['gradeSystem'].tolist()
    merged_frame['grade'] = dataset['gradeUser'].tolist()

    # merged_frame['dashboard'] = dataset['dashboard'].tolist()
    # merged_frame['favorite'] = dataset['favorite'].tolist()
    # merged_frame['clicks'] = dataset['clicks'].tolist()
    url_all = merged_frame['url']
    grades_all = merged_frame['grade'].tolist()
    grade_sys = merged_frame['gradeSystem'].tolist()
    merged_frame = merged_frame.drop(['grade'], axis=1)
    merged_frame = merged_frame.drop(['gradeSystem'], axis=1)
    merged_frame = merged_frame.drop(['url'], axis=1)
    return merged_frame, grades_all, grade_sys, url_all


def recommend_clothes(model):
    # ask query from the user
    feed = user_input_dataset()

    # get all clothes from query (rated and unrated)
    queryFeed = merged_frame.loc[feed['ProductNo'].tolist()]

    # get rated clothes from query
    ratedClothes = queryFeed.loc[queryFeed['gradeUser'] != 5]

    # get rated clothes from query
    unratedClothes = queryFeed.loc[queryFeed['gradeUser'] == 5]

    # keep grade system
    gradeSys = unratedClothes['gradeSystem']

    # keep image source
    url = unratedClothes['url']

    # delete grade user == 5
    unratedClothes = unratedClothes.drop(labels='gradeUser', axis=1)

    # delete grade system
    unratedClothes = unratedClothes.drop(labels='gradeSystem', axis=1)

    # delete image source
    unratedClothes = unratedClothes.drop(labels='url', axis=1)

    # predict ratings
    pred = model.predict(unratedClothes)

    # update dataframe with predicted ratings
    unratedClothes['predUser'] = pred

    # update dataframe with grade system
    unratedClothes['gradeSystem'] = gradeSys

    # update dataframe with image source
    unratedClothes['url'] = url

    # get well rated clothes for recommendations
    unratedClothes = unratedClothes.loc[unratedClothes['predUser'] == 1]

    # sort clothes based on grade system
    unratedClothes = unratedClothes.sort_values(by='gradeSystem', ascending=True)

    return ratedClothes, unratedClothes


def show_recommended_clothes(unratedClothes):
    # print recommendations
    for i in range(len(unratedClothes)):
        cloth = unratedClothes.iloc[i]
        rec_cloth = cloth['url']
        print('-----------------')
        image = io.imread(rec_cloth)
        plt.imshow(image)
        plt.show()


def show_liked_clothes(ratedClothes):
    # print liked clothes
    likedClothes = ratedClothes.loc[ratedClothes['gradeUser'] >= 8]
    for i in range(len(likedClothes)):
        cloth = likedClothes.iloc[i]
        rec_cloth = cloth['url']
        print('-----------------')
        image = io.imread(rec_cloth)
        plt.imshow(image)
        plt.show()


def query_accuracy(ratedClothes, model):
    gradeUser = [1 if x >= 8 else 0 for x in ratedClothes['gradeUser'].tolist()]
    pred_frame = ratedClothes
    pred_frame = pred_frame.drop(['gradeUser', 'gradeSystem', 'url'], axis=1)
    predUser = model.predict(pred_frame)
    return accuracy_score(gradeUser, predUser)

# load multi data
import pandas as pd
# read dataset
engine = helper_functions.ENGINE
dbName = helper_functions.DB_NAME
dataset = pd.read_sql_query('''SELECT PRD.Adapter, PRD.Description, PRD.Image, PRD.ImageSource, PRD.Brand, PRD.ProductCategory, PRD.ProductSubcategory, PRD.Length, PRD.Sleeve, PRD.CollarDesign, PRD.NeckDesign
                                FROM %s.dbo.PRODUCT AS PRD''' % dbName, engine)
dataset.rename(columns={'Oid': 'clothId'}, inplace=True)
dataset.insert(0,'userId',0)
columns =list(dataset.columns)
all_data = dataset
print(all_data.columns)
data = all_data.loc[all_data['userId'] == 0]

# import pandas as pd
# # read dataset
# all_data = pd.read_csv('final.csv')
# all_data
# data = all_data.loc[all_data['UserId'] == 0]
# data



# preprocessing data
## **Preprocessing Fit**

# su = data.loc[data['UserId'] == 0]
# fit_frame = lower_dimension_frame(data,['FitID'])
# len(data['FitID'].unique())
# len(fit_frame.columns)
# fit_frame


## **Preprocessing Neck Design**

# Reduced categories from 20 to 7
su = data.loc[data['userId'] == 0]
neck_frame = lower_dimension_frame(data,['NeckDesignID'])
print(len(data['NeckDesign'].unique()))
print(len(neck_frame.columns))
neck_frame

#%%

all_data['NeckDesign'].unique()

#%%

len(all_data['NeckDesign'].unique())

#%%

len(neck_frame.columns)

#%% md

## **Preprocessing Length**

#Reduced categories from 7 to 6

#%%

su = all_data.loc[all_data['userId'] == 0]
length_frame = lower_dimension_frame(su,['Length'])
length_frame

#%%

len(all_data['Length'].unique())

#%%

len(length_frame.columns)

#%% md

## **Preprocessing Product Category & SubCategory**

#Reduced categories from 179 to 33

#%%

su = all_data.loc[all_data['userId'] == 0]
# 'ProductCategory','ProductSubcategory'
prod_frame = lower_dimension_frame(su,['ProductCategory','ProductSubcategory'])
prod_frame


#%%

len(data['ProductSubcategory'].unique())+ len(data['ProductCategory'].unique())

#%%

len(prod_frame.columns)

#%% md

## **Preprocessing Search-Word**

#%%

su = all_data.loc[all_data['UserId'] == 0]
single_user_data = su.copy()
unique_search_words = single_user_data['SearchWords'].unique()
future_frame = []
for i in single_user_data['ImageSource'].tolist():
  z_array = np.zeros((1,len(unique_search_words)))
  local_sw = su.loc[su['ImageSource'] == i]['SearchWords'].tolist()
  #print(local_sw)
  for j in range(len(unique_search_words)):
    for k in range(len(local_sw)):
      if local_sw[k] == unique_search_words[j]:
        #print(local_sw[k])
        z_array[0][j] = 1
  future_frame.append(z_array)


#%%

future_frame = np.array(future_frame).squeeze()
search_word_Dataframe = pd.DataFrame(data=future_frame,columns=unique_search_words)
index = prod_frame.index.tolist()
search_word_Dataframe['index'] = index
search_word_Dataframe = search_word_Dataframe.set_index(keys='index')
np.array(search_word_Dataframe.columns.tolist())

#%%

search_word_Dataframe.columns.tolist()

#%%

len(search_word_Dataframe)

#%% md

# merge

#%%

merged = pd.merge(prod_frame,length_frame,on='index')
merged = pd.merge(merged,search_word_Dataframe,on='index')
#merged = pd.merge(merged,fit_frame,on='index')
merged = pd.merge(merged,neck_frame,on='index')
merged

#%%

dataset = data.drop_duplicates(subset='ImageSource',keep='first')
#dataset = dataset.loc[dataset['gradeUser'] !=2.5]
grades = dataset['GradeByUser'].tolist()
index = dataset['ProductNo'].tolist()
url = np.array(dataset['ImageSource'].tolist())
merged = merged.loc[index]
merged = merged.reset_index()
merged = merged.drop(columns='index')
merged

print("kdkd")