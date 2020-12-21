import seaborn as sns
sns.set_theme()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


class evaluator:

    def calculate_metrics(self, df, threshold):
        y_true = df['y_true'].tolist()
        y_pred = df['y_pred'].tolist()

        # calculate rmse
        rmse = mean_squared_error(y_true, y_pred) ** (1 / 2)

        # calculate mae
        mae = mean_absolute_error(y_true, y_pred)
        # convert predictions to binary based on threshold

        pred = [1 if x >= threshold else 0 for x in y_pred]
        true = [1 if x >= threshold else 0 for x in y_true]
        # calculate accuracy
        acc_ = accuracy_score(true, pred)
        # calculate recall
        rec_ = recall_score(true, pred)
        # calculate precision
        prec_ = precision_score(true, pred)
        # calculate f1 score
        f1_ = f1_score(true, pred)

        return rmse, mae, acc_, rec_, prec_, f1_

    def nei_metric_df(self, user, neigh, df, threshold):
        # calculate matrics
        rmse, mae, acc_, rec_, prec_, f1_ = self.calculate_metrics(df, threshold)
        # create a dataframe with those metrics, scores and neighbors
        df = pd.DataFrame(columns=['user', 'metric', 'score', 'neighbors'])
        # append accuracy
        df = df.append(
            pd.DataFrame(data=[[user, 'accuracy', acc_, neigh]], columns=['user', 'metric', 'score', 'neighbors']))
        # append recall
        df = df.append(
            pd.DataFrame(data=[[user, 'recall', rec_, neigh]], columns=['user', 'metric', 'score', 'neighbors']))
        # append precision
        df = df.append(
            pd.DataFrame(data=[[user, 'precision', prec_, neigh]], columns=['user', 'metric', 'score', 'neighbors']))
        # append f1_score
        df = df.append(
            pd.DataFrame(data=[[user, 'f1_score', f1_, neigh]], columns=['user', 'metric', 'score', 'neighbors']))
        df2 = pd.DataFrame(columns=['user', 'metric', 'score', 'neighbors'])
        # append accuracy
        df2 = df2.append(
            pd.DataFrame(data=[[user, 'rmse', rmse, neigh]], columns=['user', 'metric', 'score', 'neighbors']))
        # append recall
        df2 = df2.append(
            pd.DataFrame(data=[[user, 'mae', mae, neigh]], columns=['user', 'metric', 'score', 'neighbors']))

        return df, df2

    def user_metric_df(self, user, df, threshold):
        # calculate matrics
        rmse, mae, acc_, rec_, prec_, f1_ = self.calculate_metrics(df, threshold)
        # create a dataframe with those metrics and scores
        df = pd.DataFrame(columns=['user', 'metric', 'score'])
        # append accuracy
        df = df.append(pd.DataFrame(data=[[user, 'accuracy', acc_]], columns=['user', 'metric', 'score']))
        # append recall
        df = df.append(pd.DataFrame(data=[[user, 'recall', rec_]], columns=['user', 'metric', 'score']))
        # append precision
        df = df.append(pd.DataFrame(data=[[user, 'precision', prec_]], columns=['user', 'metric', 'score']))
        # append f1_score
        df = df.append(pd.DataFrame(data=[[user, 'f1_score', f1_]], columns=['user', 'metric', 'score']))

        df2 = pd.DataFrame(columns=['user', 'metric', 'score'])
        # append accuracy
        df2 = df2.append(pd.DataFrame(data=[[user, 'rmse', rmse]], columns=['user', 'metric', 'score']))
        # append recall
        df2 = df2.append(pd.DataFrame(data=[[user, 'mae', mae]], columns=['user', 'metric', 'score']))

        return df, df2

    def visualize_bars(self, df, axe_x, axe_y, hue):
        g = sns.catplot(
            data=df, kind="bar",
            x=axe_x, y=axe_y, hue=hue,
            ci="sd", palette="dark", alpha=.6, height=6, legend_out=True
        )
        g.despine(left=True)
        g.set_axis_labels("", "Percent (%)")
        g.legend.set_title("")