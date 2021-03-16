# import all libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
'''
@author: Mpountou
@year: 2020
'''

class data_handler:
  """
    A class to handle and preprocess data before make recommendation
    ...

    Attributes
    ----------
    path : str
       location directory file of data
    columns : array
       array of columns that should keep from data

    Methods
    -------
    loadDataFrame()
       retruns dataframe based on path and columns
    deep_preprocessing()
       returns dataframe for deep learning model based, total users, total items and minmax of ratings
    split()
       splits the data given the user input
    create_matrix()
       creates a user - item matrix given the user item dataframe

    """
  def __init__(self,path,columns):
    # make path global
    self.path = path
    # make cols global
    self.columns = columns

  def loadDataFrame(self):
    # generate dataframe - user,item,rating
    dataset = pd.read_pickle(self.path)

    # return data with declared columns
    return dataset[self.columns]

  def deep_preprocessing(self,dataset):
    # init encoder
    encoder = LabelEncoder()
    # create new data frame for deep learning model
    df = pd.DataFrame(columns=dataset.columns)
    # user encode
    df[columns[0]] = encoder.fit_transform(dataset[columns[0]].values)
    # get num of total users
    t_users = df[columns[0]].nunique()
    # init encoder
    itemencoder = LabelEncoder()
    # item encode
    df[columns[1]] = itemencoder.fit_transform(dataset[columns[1]].values)
    # get num of total items
    t_items = df[columns[1]].nunique()
    # typecast rating to float
    df[columns[2]] = dataset[columns[2]].values.astype(np.float32)
    # find min and max of ratings
    minmax = (min(df[columns[2]]),max(df[columns[2]]))

    # return dataframe,total users, total items, and min-max of ratings
    return df,t_users,t_items,minmax,itemencoder,encoder

  def split(self,df,input_user,test_size):
    # get ratings of input user
    input_user_data = df.loc[df[self.columns[0]] == input_user]
    # split ratings to test and train
    train_X, test_X, train_y, test_y = train_test_split(input_user_data, input_user_data.index.tolist(), test_size=test_size, random_state=1)
    # copy all data
    tmp_dataset = df.copy()
    # remove test data
    tmp_dataset.loc[test_X.index.tolist(),self.columns[2]] = -1
    # keep all data except of test data
    tmp_dataset = tmp_dataset.loc[tmp_dataset[self.columns[2]] >=0]
    # declare train data
    train_X = tmp_dataset

    # return train and test data
    return train_X,test_X

  def rec_split(self,df):
    train_X, test_X, train_y, test_y = train_test_split(df, df.index.tolist(), test_size=0.01, random_state=1)
     # return train and test data
    return train_X,test_X

  def create_matrix(dataset,columns,fill_unrated_with):
    # unique value of every user
    users = dataset[columns[0]].unique()
    # number of total users
    t_users = len(users)
    # unique value of every items
    items = dataset[columns[1]].unique()
    items.sort()
    # number of total items
    t_items = len(items)
    # initialize data with zeros
    data = np.empty((t_users,t_items))
    data[:] = fill_unrated_with
    # create user - item matrix
    matrix = pd.DataFrame(data= data,columns=items)

    # fill user-item matrix with ratings
    for user in range(len(users)):
      # current user dataframe
      user_ = dataset.loc[dataset[columns[0]] == users[user]]
      # every item id that user rated
      itemID = user_[columns[1]].tolist()
      # every rating for every item that user rated
      ratingValue = user_[columns[2]].tolist()
      for j in range(len(itemID)):
        # fill ratings on user-item matrix
        matrix[itemID[j]][user] =  ratingValue[j]

    # return user-item matrix
    return matrix