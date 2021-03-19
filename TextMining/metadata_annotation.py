import pandas as pd
import numpy as np
import sqlalchemy
import sys
import pymssql
import itertools
from collections import defaultdict
from difflib import SequenceMatcher
import time
import logging

from core.helper_functions import *
import core.config as config
from core.logger import S4F_Logger

def updateQuery(df, engine):
    ###Create temp table and then update only the rows that belong to temp AND product###
    df.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
    # df.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(config.UPDATESQLQUERY)

#Relevant similarity function#

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def contains_word(s, w):
    return f' {w} ' in f' {s} '

def editStrings(s):
    ### Check for specific values of s and change the values to match those of Energiers ###
    s = (s[0], 'V NECK', s[2], s[3]) if (s[1] == 'V-NECK' and attr == 'NeckDesign') else s
    s = (s[0], 'OFF SHOULDER', s[2], s[3]) if (s[1] == 'OFF-SHOULDER' and attr == 'NeckDesign') else s
    s = (s[0], s[1] + " LENGTH", s[2], s[3]) if (s[1] == 'SHORT' or s[1] == 'MEDIUM' or s[1] == 'KNEE' and attr == 'Length') else s
    s = (s[0], 'LONG', s[2], s[3]) if (s[1] == 'MAXI' and attr == 'Length') else s
    s = (s[0], s[1] + " COLLAR", s[2], s[3]) if ((s[1] == 'MAO' or s[1] == 'STAND UP' or s[1] == 'POLO') and attr == 'CollarDesign') else s
    s = (s[0], s[1] + " FIT", s[2], s[3]) if ((s[1] == 'REGULAR' or s[1] == 'RELAXED' or s[1] == 'SLIM') and attr == 'Fit') else s
    return s


if __name__ == "__main__":
    # Begin Counting Time
    start_time = time.time() 
    user = sys.argv[1]
    logger = S4F_Logger('TextAnnotationLogger').logger
    helper = Helper(logger)

    try:
        ### Read Table Products from S4F database ###
        logger.info('Loading Product table...')
        #Connect to database with sqlalchemy
        currendDir = config.TEXT_MINING
        engine = config.ENGINE
        dbName = config.DB_NAME

        productsDF = pd.read_sql_query('SELECT * FROM  %s.dbo.Product' % dbName, engine)
        # productsDF = pd.read_sql_query('SELECT * FROM public.\"Product\"', engine)

        labelsDF = pd.DataFrame()
        
        ### Filter the old elements with the new one, Keep only the non-updated elements to improve efficiency, reset index due to slice ###
        # productsDF = productsDF[productsDF.loc[:, (config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTESID)].isnull().apply(lambda x: all(x), axis=1)]
        productsDF = productsDF[productsDF.loc[:, config.PRODUCT_ATTRIBUTES].isnull().apply(lambda x: all(x), axis=1)]
        productsDF = productsDF.reset_index(drop=True)
        # productsDF = productsDF.drop(config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTESID, axis = 1)
        labelsDF['Oid'] = productsDF['Oid'].copy()
        ### Metadata and Headline consists of information related to each row ###

        metadata = productsDF['Metadata'].str.upper()
        headline = productsDF['Description'].str.upper()

        ### Read possible labels###

        expertAttributesDF = pd.read_excel(config.PRODUCT_ATTRIBUTES_PATH, sheet_name=config.SHEETNAME)
        
        # Create Variables with same name as the Energiers column names, to store the labels. Create new columns at assos with Energiers column names #
        attrDict = {}
        for attr in config.PRODUCT_ATTRIBUTES:
            attrDict[str(attr)] = list(expertAttributesDF[attr].replace(' ', np.nan).dropna().unique())
            attrDict[str(attr)] = [la.upper() for la in attrDict[str(attr)]]
            labelsDF[attr] = np.empty((len(productsDF), 0)).tolist()
        
        ### Preprocessing ###
        logger.info('Preprocessing metadata...')    
        # Convert every label, metadata and headline to uppercase #
        splitted_metadata = [s.split() if isinstance(s,str) else " " for s in metadata]
        splitted_headline = [s.split() if isinstance(s,str) else " " for s in headline]    
        
        ### Search for occurences of labels in metadata and headline. For category length if the next 
        #   word is not a kind of cat (Trousers etc) then it is propably wrong so we get rid of it. ###
        cat = 'ProductCategory'
        # for attr in (config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTES):
        for attr in config.PRODUCT_ATTRIBUTES:
            saved_meta = [(index, label, s.find(label), 1) for label in (attrDict[str(attr)]) 
                    for index,s in enumerate(metadata) if contains_word(s, label)]
            saved_head = [(index, label, s.find(label), 0) for label in (attrDict[str(attr)]) 
                    for index,s in enumerate(headline) if contains_word(s, label)]
            for s in (saved_meta + saved_head):
                s = editStrings(s)
                labelsDF.loc[s[0],attr].append((s[1], s[2], s[3]))
                if attr == 'Length':
                    flag = 0
                    for i,strSen in enumerate(splitted_metadata+splitted_headline):
                        for j,sen in enumerate(strSen[:-1]):
                            if sen == s[1] and 'SLEEVE' == strSen[j+1] and flag == 0:
                                flag = 1             
                    if flag == 1:
                        labelsDF.loc[s[0], attr].remove((s[1], s[2], s[3]))
        
        ### Find similar words, for example -> rounded and round and one of them is discarded ###
        # for attr in (config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTES):
        for attr in config.PRODUCT_ATTRIBUTES:
            # Τhe next line sorts the elements of the respective columns based on position and on metadata or headline (headline first and then position on each string)
            labelsDF.loc[:, attr] = pd.Series([sorted(list(ele), key = lambda tup: (tup[2], tup[1])) for ele in labelsDF.loc[:, attr]])
            labelsDF.loc[:, attr] = pd.Series([list(map(lambda x: x[0], ele)) for ele in labelsDF.loc[:, attr]])
            saved = defaultdict(list)
            for i, element in enumerate(labelsDF.loc[:, attr]):
                if len(element) >= 2:
                    for (k, l1),(ki, l2) in itertools.combinations(enumerate(element), 2): 
                        if similar(l1[0], l2[0]) >= 0.8:
                            saved[i].append((ki,l2))

            for key,value in saved.items():
                uni = np.unique(value, axis = 0)
                for index, x in uni:
                    # We reverse because remove always pop outs the first element while we want the last
                    labelsDF.loc[key, attr].reverse()
                    labelsDF.loc[key, attr].remove(x)
                    labelsDF.loc[key, attr].reverse()

        ### Check if list is empty and in this case make it None ###
        # for attr in (config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTES):
        for attr in config.PRODUCT_ATTRIBUTES:
            labelsDF.loc[:, attr] = labelsDF[attr].apply(lambda x: ','.join(x) if x else None)
        #Extract unique labels from metadata and headline
        labelsUnique = {attr:set([l for label in labelsDF[attr].unique() if label for l in label.split(',')]) for attr in labelsDF.loc[:, config.PRODUCT_ATTRIBUTES].columns}

        # Read from database the labels 
        # for name in (config.PRODUCT_ATTRIBUTES + config.DEEPFASHIONATTRIBUTES):
        dfDict = {}
        for attr in config.PRODUCT_ATTRIBUTES:
            dfDict[str(attr)+'_DB'] = pd.read_sql_query("SELECT * FROM %s.dbo.%s" % (dbName, attr), engine)
            # dfDict[str(attr)+'_DB'] = pd.read_sql_query(' SELECT * FROM public.\"%s\"' % str(attr), engine)

        # Update the DB label tables with the new attributes    
        for (label, values) in labelsUnique.items():
            for v in values:
                if v not in dfDict[label + '_DB']['Description'].values:
                    if label=='ProductSubcategory':
                        submitdf = pd.DataFrame([{'Description': v, 'AlternativeDescription': '', 'ProductCategory': None, 
                                                'Active': True, 'OptimisticLockField': None}])
                        submitdf.to_sql(label, schema='%s.dbo' % dbName, con=engine, if_exists='append', index=False)
                        # submitdf.to_sql(label, con=engine, if_exists='append', index=False)
                    else:
                        submitdf = pd.DataFrame([{'Description': v, 'AlternativeDescription': '', 'Active': True,
                                                'OptimisticLockField': None}])
                        submitdf.to_sql(label, schema='%s.dbo' % dbName, con=engine, if_exists='append', index=False)
                        # submitdf.to_sql(label, con=engine, if_exists='append', index=False)
            
        logger.info('Update product attributes')  
        ## Update Product table with the foreign key values of the updated attributes
        # re-load from database the updated attribute tables and create a dataframe for each 
        dfDict = {}
        for attr in config.PRODUCT_ATTRIBUTES:
            dfDict[str(attr)+'_DB'] = pd.read_sql_query(''' SELECT * FROM %s.dbo.%s''' % (dbName, str(attr)), engine)
            # dfDict[str(attr)+'_DB'] = pd.read_sql_query(''' SELECT * FROM public.\"%s\"''' % str(attr), engine)

        # Update Products table for each attribute
        for attr in config.PRODUCT_ATTRIBUTES:
            # If there are multiple attributes, select the first
            labelsDF.loc[labelsDF[attr].notnull(), attr] = labelsDF[labelsDF[attr].notnull()][attr].apply(lambda x: x.split(',')[0] if x else None)
            # Merge the updated attribute values to Product table
            mergedDF = labelsDF.merge(dfDict[str(attr)+'_DB'], left_on=attr, right_on='Description')[['Oid_x', 'Oid_y']]
            productsDF.loc[productsDF['Oid'].isin(mergedDF['Oid_x'].values), attr] = mergedDF['Oid_y'].values
        # Execute Product update query
        productsDF.to_sql("temp_table", schema='%s.dbo' % dbName, con=engine, if_exists='replace', index=False)
        # productsDF.to_sql("temp_table", con = engine, if_exists = 'replace', index = False)
        with engine.begin() as conn:
            conn.execute(config.UPDATESQLQUERY)

        logger.info("--- %s seconds ---" % (time.time() - start_time))
    except Exception as ex:
        logger.warn_and_exit(ex)

