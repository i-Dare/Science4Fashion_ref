import pandas as pd
from evaluator import evaluator
from ContentBasedRecom import cb_rf
from core.helper_functions import Helper
eval = evaluator()
currendDir = Helper.WEB_CRAWLERS
engine = Helper.ENGINE
dbName = Helper.DB_NAME

df = pd.DataFrame(columns=['user','metric','score'])
df2 = pd.DataFrame(columns=['user','metric','score'])
u = 0
# for u in range(10):
model = cb_rf(input_user=u, engine=engine)
grade_df = model.split_and_predict()
a,b = eval.user_metric_df(user=u,df=grade_df,threshold=3)
df = df.append(a)
df2 = df2.append(b)

eval.visualize_bars(df = df,axe_x='metric',axe_y='score',hue='user')

eval.visualize_bars(df = df2,axe_x='metric',axe_y='score',hue='user')