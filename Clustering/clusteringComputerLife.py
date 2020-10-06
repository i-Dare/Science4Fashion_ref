import pyodbc
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn import metrics
import sqlalchemy
import constants

#X = pd.DataFrame(columns = ['Price'] + constants.ATTRIBUTESSOCIALMEDIA[1:])
X = np.empty((0, len(constants.ATTRIBUTESSOCIALMEDIA)))
engine = {}
#Iter through dbs
for name in constants.DBNAMES:
    engine[name] = sqlalchemy.create_engine(constants.CONNECTION + name)
    SQL_Query = pd.read_sql_query(constants.SELECTSQLQUERY.format(name), engine[name])
    df = pd.DataFrame(SQL_Query)
    df = df.fillna(0)
    if name == 'SocialMedia':
        sth = df[constants.ATTRIBUTESSOCIALMEDIA] #'Inspiration_Background'
        ###Read all the tables of features from SocialMedia DB to match id to genID###
        for names in (constants.MIDTABLESSOCIALMEDIA):
            locals()[str(names)+'_DB'] = pd.read_sql_query(constants.SELECTQUERYMID + str(names.upper()), engine[name])

        ### Merging mid tables with product to create the product with values not id.
        for col in (constants.MIDTABLESSOCIALMEDIA):
            testing = sth.merge(locals()[str(col)+'_DB'], how = 'left', on=(str(col)+'ID'))
            sth.loc[:,str(col)+'ID'] = testing.loc[:,'Gen' + str(col) + 'ID']
            # labelsDF.loc[:,str(col)] = testing.loc[:,str(col)]
        ProductSo = df['ProductNo']
    else:
        sth = df[constants.ATTRIBUTESNRGSEARCH]
        ProductNo = df['ProductNo']
        socialLen = X.shape[0]

    sth = sth.to_numpy()
    sth[:, 1:] = sth[:,1:].astype('int')
    #X contains elements from both social media and nrgsearch
    X = np.append(X, sth, axis = 0)
    
# Clustering process using 6 clusters
kproto = KModes(n_clusters=6, init=constants.INITKMODES, verbose=2)
clusters = kproto.fit_predict(X,  categorical=[1, 2, 3, 4, 5, 6, 7, 8, 9])
center = kproto.cluster_centroids_
centerNew = np.zeros((6,11))
centerNew[:,0] = range(6)
centerNew[:,1:5] = center[:,:4]
centerNew[:,5] = -1
centerNew[:,6:] = center[:,4:]
centerNew = pd.DataFrame(centerNew)
#print(centerNew.iloc[:, 2:].astype('int'))

# Center of each cluster
centerNew[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]] = centerNew[[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]].astype('int')
centerNew.rename(columns=constants.COLUMNS, inplace=True)


# Saving clusters to db and updating centers
for name in constants.DBNAMES:
    use = 'ProductNo'
    centerNew.to_sql("temp_table", schema='dbo', con=engine[name], if_exists='replace', index=False)
    with engine[name].begin() as conn:
        conn.execute(constants.UPDATESQLQUERY)
    #IT DEPENDS ON THE DATABASE
    if name == 'SocialMedia':
        data = pd.DataFrame({'ProductNo': ProductSo, 'ClusterID': clusters[:socialLen]})
    else:
        data = pd.DataFrame({'ProductNo': ProductNo, 'ClusterID': clusters[socialLen:]})

    data.to_sql("temp_table", schema='dbo', con=engine[name], if_exists='replace', index=False)
    with engine[name].begin() as conn:
        conn.execute(constants.UPDATESQLQUERY2.format(use,use,use,use))

# Print training statistics
# print(kproto.cost_)
# print(kproto.n_iter_)

# Save to db:   a) a table with the product id and the cluster id: < df[['Product_No']], clusters >
#               b) a table with 6 rows (equal to the number of clusters) with the cendroids: < kproto.cluster_centroids_ >




