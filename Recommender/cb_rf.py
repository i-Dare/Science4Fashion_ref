from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import optuna
import sklearn
import numpy as np
from matplotlib import pyplot

from evaluator import evaluator

class cb_rf():

  def __init__(self,input_user,features, dataset, columns, wholedata):
    self.evdf = -1

    self.features = features

    self.dataset = dataset

    self.columns = columns

    self.recdf = -1

    self.input_user = input_user

    self.wholedata = wholedata


  def make_recommendation(self):

    if type(self.recdf) == type(pd.DataFrame()):
      return self.recdf

    # make local variables
    dataset = self.dataset
    columns = self.columns
    features = self.features
    wholedata = self.wholedata

    # get unrated ratings per user
    all_users = wholedata[columns[0]].unique()
    all_items = set(wholedata[columns[1]].unique())

    # dataframe for recommendation
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
    for i in all_users:
      u_data = dataset.loc[dataset[columns[0]] == i]
      train_X = features.loc[u_data[columns[1]].tolist()]

      train_y = u_data[columns[2]].tolist()
      self.t_x = train_X
      self.t_y = train_y

      study = optuna.create_study(direction='minimize')
      study.optimize(self.m_objective, n_trials=10)
      trial_m = study.best_trial
      regr = RandomForestRegressor(n_estimators=trial_m.params['n_estimators'],max_depth=trial_m.params['max_depth'],criterion='mse', random_state=0)
      regr.fit(train_X,train_y)

      rated = set(u_data[columns[1]].unique())
      unrated = list(all_items - rated)
      if len(unrated)>0:
        test_X = features.loc[unrated]
        y_rec = regr.predict(test_X)
        p_data = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
        p_data[columns[0]] = [int(i) for k in range(len(unrated))]
        p_data[columns[1]] = unrated
        p_data['y_rec'] = y_rec
        df = df.append(pd.DataFrame(data=p_data.values,columns=[columns[0],columns[1],'y_rec']))

    df = df.reset_index().drop(columns='index')

    self.recdf = df

    return df

  def m_objective(self,ttrial):
    t_x = self.t_x
    t_y = self.t_y

    n_estimators = ttrial.suggest_int('n_estimators', 30, 40)
    max_depth = int(ttrial.suggest_loguniform('max_depth', 20, 32))
    clf = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(t_x, t_y, random_state=0)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)
    return error
    #return sklearn.model_selection.cross_val_score(clf, t_x.values,t_y,
    #      n_jobs=-1, cv=3).mean()

  def evaluate_system(self,all_train,all_test):
    if type( self.evdf) == type(pd.DataFrame()):
      return  self.evdf
    # make local variables
    dataset = self.dataset
    columns = self.columns
    features = self.features

    all_users = dataset[columns[0]].unique()

    df = pd.DataFrame(columns=[columns[0],columns[1],'y_true','y_pred'])

    for i in all_users:

      u_train = all_train.loc[all_train[columns[0]] == i]
      u_test = all_test.loc[all_test[columns[0]] == i]

      f_train = features.loc[u_train[columns[1]].tolist()]
      f_test = features.loc[u_test[columns[1]].tolist()]

      train_y = u_train[columns[2]].tolist()
      test_y = u_test[columns[2]].tolist()

      self.t_x = f_train
      self.t_y = train_y


      study = optuna.create_study(direction='minimize')
      study.optimize(self.m_objective, n_trials=10)
      trial_m = study.best_trial
      regr = RandomForestRegressor(n_estimators=trial_m.params['n_estimators'],max_depth=trial_m.params['max_depth'],criterion='mse', random_state=0)
      regr.fit(f_train,train_y)
      y_pred = regr.predict(f_test)

      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_true','y_pred'])
      u_df[columns[0]] = u_test[columns[0]].tolist()
      u_df[columns[1]] = u_test[columns[1]].tolist()
      u_df['y_true'] = u_test[columns[2]].tolist()
      u_df['y_pred'] = y_pred

      df = df.append(u_df)

    df = df.reset_index().drop(columns='index')
    self.evdf = df
    return df



  # def recommend(self,itemsTopredict):
  #   # init regressor rf
  #   regr = RandomForestRegressor(n_estimators=100,criterion='mse', random_state=1)
  #   # train
  #   regr.fit(self.rated_clothes, self.ratings)
  #   # predict
  #   pred_X = self.unrated_clothes.loc[itemsTopredict]
  #   y_pred = regr.predict(pred_X)
  #   # result dataframe
  #   df = pd.DataFrame()
  #   df['clothId'] = pred_X.index.tolist()
  #   df['rf_pred'] = y_pred
  #
  #   return df
  #
  # def split_and_predict(self):
  #   from sklearn.ensemble import RandomForestClassifier
  #   # init regressor rf
  #   regr = RandomForestRegressor(n_estimators=15,max_depth=5,criterion='mse', random_state=0)
  #   #regr = RandomForestClassifier(n_estimators=15,max_depth=5, random_state=0)
  #   # split data
  #   train_X, test_X, train_y, test_y = train_test_split(self.rated_clothes, self.ratings, test_size=0.1, random_state=0)
  #   # train
  #   regr.fit(train_X,train_y)
  #   # predict
  #   y_pred = regr.predict(test_X)
  #
  #   # result dataframe
  #   df = pd.DataFrame(columns=['clothId','y_pred','y_true'])
  #   df['clothId'] = test_X.index.tolist()
  #   df['y_true'] = test_y
  #   df['y_pred'] = y_pred
  #
  #   return df
  # def hyb_eval(self,train,test):
  #   # init regressor rf
  #   regr = RandomForestRegressor(n_estimators=100,criterion='mse', random_state=1)
  #   # train
  #   d_train = self.rated_clothes.copy()
  #   d_train['rating'] = self.ratings.copy()
  #   train_X = d_train.loc[train['clothId'].tolist()]
  #   train_y = train_X['rating'].tolist()
  #   train_X = train_X.drop(columns='rating')
  #   d_test = self.rated_clothes.copy()
  #   d_test['rating'] = self.ratings
  #   test_X = d_test.loc[test['clothId'].tolist()]
  #   test_y = test_X['rating'].tolist()
  #   test_X = test_X.drop(columns='rating')
  #   # train
  #   regr.fit(train_X,train_y)
  #   # predict
  #   y_pred = regr.predict(test_X)
  #
  #   # result dataframe
  #   df = pd.DataFrame(columns=['clothId','rf_pred'])
  #   df['clothId'] = test_X.index.tolist()
  #   #df['y_true'] = test_y
  #   df['rf_pred'] = y_pred
  #
  #   return df
  #
  # def coverage(self,threshold):
  #
  #   if type(self.recdf) != type(pd.DataFrame()):
  #     pred_ratings = self.make_recommendation()
  #   else:
  #     pred_ratings = self.recdf
  #
  #   already_rated = len(self.dataset)
  #
  #   high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>threshold])
  #
  #   low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=threshold])
  #
  #   unrated = len(pred_ratings.loc[pred_ratings['y_rec']==np.nan])
  #
  #   cov_df = pd.DataFrame()
  #
  #   cov_df['recommended'] = [high_rated]
  #
  #   cov_df['not recommended'] = [low_rated]
  #
  #   cov_df['cannot recommended'] = [unrated]
  #
  #   cov_df['already rated'] = [already_rated]
  #
  #   return cov_df
  #
  #
  #
  # def novelty(self,cat_,threshold,translator_=False):
  #
  #   if type(self.recdf) != type(pd.DataFrame()):
  #     pred_ratings = self.make_recommendation()
  #   else:
  #     pred_ratings = self.recdf
  #
  #   pred_ratings = pred_ratings.merge(cat_,on=columns[1],how='inner')
  #
  #   categories = pred_ratings['category'].unique()
  #
  #   c_ratings = []
  #   for i in range(len(categories)):
  #     ratings = []
  #     fr = pred_ratings.loc[pred_ratings['category'] == categories[i]]
  #     ratings.append(round(len(fr.loc[fr['y_rec'] >=threshold])  / len(fr.loc[fr['y_rec'] >=0]),2) )
  #     ratings.append(round(len(fr.loc[fr['y_rec'] <threshold])  / len(fr.loc[fr['y_rec'] >=0]) ,2))
  #     c_ratings.append(ratings)
  #
  #   df = pd.DataFrame(data=c_ratings , columns=['προτείνεται','δεν προτείνεται'])
  #   if type(translator_) == bool:
  #     return df
  #
  #   categories_gr = []
  #
  #   for i in range(len(categories)):
  #     categories_gr.append(translator_.loc[translator_['category'] == categories[i]].index.tolist()[0])
  #   df['κατηγορίες'] = categories_gr
  #
  #   df = df.set_index(keys='κατηγορίες')
  #
  #   return df
