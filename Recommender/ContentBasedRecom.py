from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import evaluator


class cb_rf():

    def __init__(self, input_user,engine):
        self.input_user = input_user
        self.ASK_SQL_Query = pd.read_sql_query('''SELECT * FROM S4F.dbo.Product''', engine)
        self.df = pd.DataFrame(self.ASK_SQL_Query)
        #self.df = pd.read_pickle('/content/drive/My Drive/multi-user5.pkl')
        self.df = self.df.loc[self.df['UserId'] == input_user]
        self.df = self.df.drop_duplicates(subset='ImageSource', keep='first')
        self.prodNo = self.df['ProductNo'].tolist()
        self.ratings = self.df['gradeUser'].tolist()
        self.all_clothes = pd.read_pickle('/content/drive/My Drive/clothes_attr.pkl')
        self.rated_clothes = self.all_clothes.copy()
        self.rated_clothes['ratings'] = self.ratings

        z = []
        for i in range(4325):
            z.append([])

        for i in range(10):
            if i != self.input_user:
                rating_data = pd.read_pickle('/content/drive/My Drive/multi-user5.pkl')
                u_data = rating_data.loc[rating_data['UserId'] == i]
                u_data = u_data.drop_duplicates(subset='ImageSource', keep='first')
                for j in range(len(u_data)):
                    if u_data.iloc[j]['gradeUser'] != 2.5:
                        z[j].append(u_data.iloc[j]['gradeUser'])
        for i in range(len(z)):
            if len(z[i]) > 0:
                z[i] = sum(z[i]) / len(z[i])
            else:
                z[i] = -1
        self.rated_clothes['avg_rating'] = z
        self.unrated_clothes = self.rated_clothes.copy()
        self.unrated_clothes = self.unrated_clothes.loc[self.unrated_clothes['ratings'] == 2.5]
        self.rated_clothes = self.rated_clothes.loc[self.rated_clothes['ratings'] != 2.5]
        self.ratings = self.rated_clothes['ratings'].tolist()
        self.rated_clothes = self.rated_clothes.drop(columns='ratings')
        self.unrated_clothes = self.unrated_clothes.drop(columns='ratings')

    def recommend(self, itemsTopredict):
        # init regressor rf
        regr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1)
        # train
        regr.fit(self.rated_clothes, self.ratings)
        # predict
        pred_X = self.unrated_clothes.loc[itemsTopredict]
        y_pred = regr.predict(pred_X)
        # result dataframe
        df = pd.DataFrame()
        df['clothId'] = pred_X.index.tolist()
        df['rf_pred'] = y_pred

        return df

    def split_and_predict(self):
        # init regressor rf
        regr = RandomForestRegressor(n_estimators=80, max_depth=20, criterion='mse', random_state=1)
        # split data
        train_X, test_X, train_y, test_y = train_test_split(self.rated_clothes, self.ratings, test_size=0.1,
                                                            random_state=1)
        # train
        regr.fit(train_X, train_y)
        # predict
        y_pred = regr.predict(test_X)

        # result dataframe
        df = pd.DataFrame(columns=['clothId', 'y_pred', 'y_true'])
        df['clothId'] = test_X.index.tolist()
        df['y_true'] = test_y
        df['y_pred'] = y_pred

        return df

    def hyb_eval(self, train, test):
        # init regressor rf
        regr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1)
        # train
        d_train = self.rated_clothes.copy()
        d_train['rating'] = self.ratings.copy()
        train_X = d_train.loc[train['clothId'].tolist()]
        train_y = train_X['rating'].tolist()
        train_X = train_X.drop(columns='rating')
        d_test = self.rated_clothes.copy()
        d_test['rating'] = self.ratings
        test_X = d_test.loc[test['clothId'].tolist()]
        test_y = test_X['rating'].tolist()
        test_X = test_X.drop(columns='rating')
        # train
        regr.fit(train_X, train_y)
        # predict
        y_pred = regr.predict(test_X)

        # result dataframe
        df = pd.DataFrame(columns=['clothId', 'rf_pred'])
        df['clothId'] = test_X.index.tolist()
        # df['y_true'] = test_y
        df['rf_pred'] = y_pred

        return df

    def coverage(self):
        # init regressor rf
        regr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1)
        # train
        regr.fit(self.rated_clothes, self.ratings)
        # predict
        y_pred = regr.predict(self.unrated_clothes)
        # save pred
        df = pd.DataFrame(columns=['clothId', 'rf_pred'])
        df['clothId'] = self.unrated_clothes.index.tolist()
        df['rf_pred'] = y_pred
        low_rated = len(df.loc[df['rf_pred'] < 3])
        high_rated = len(df.loc[df['rf_pred'] >= 3])
        unrated = 1
        cov_df = pd.DataFrame()
        cov_df['high_rated'] = [high_rated]
        cov_df['low_rated'] = [low_rated]
        cov_df['unrated'] = [unrated]
        cov_df['coverage'] = [round((high_rated + low_rated) / unrated, 2)]
        return cov_df

