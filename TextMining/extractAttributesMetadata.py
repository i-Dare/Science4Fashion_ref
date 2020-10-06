import constants
import pandas as pd
import numpy as np
import sqlalchemy
import pymssql
import itertools
from collections import defaultdict
from difflib import SequenceMatcher
import time
start_time = time.time()


#Relevant similarity function#

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def contains_word(s, w):
    return f' {w} ' in f' {s} '

def connectDatabase():
    ###Establish connection with the database and execute the selection query####
    engine = sqlalchemy.create_engine(constants.CONNECTION)
    query_db = pd.read_sql_query(constants.SELECTSQLQUERY, engine)
    return engine,query_db

def preprocessDeepfashion():
    ###Read possible labels from Deepfashion###
    labelsDEEP = pd.read_csv(constants.DEEPFASHIONPATH, sep=r"[ ]{2,}", skiprows = 1, engine = 'python')
    ###Create deepfashion DF with labels and their respective category Fabric, part etc.###
    deepfashionNamesDF = pd.DataFrame({'attribute_type':range(1,6), 'attribute_group_name':constants.DEEPFASHIONATTRIBUTES})
    deepfashionLabelsDF = pd.merge(labelsDEEP, deepfashionNamesDF, on = 'attribute_type').replace(' ', np.nan).dropna()
    ###Create Variables with the same name as the Deepfashion names. Create new columns at assos with Deepfashion column names###
    groups = deepfashionLabelsDF.groupby('attribute_group_name')['attribute_name']
    return groups

def editStrings(s):
    ###Check for specif values of s and change the values to match those of Energiers###
    s = (s[0], 'V NECK', s[2], s[3]) if (s[1] == 'V-NECK' and col == 'NeckDesign') else s
    s = (s[0], 'OFF SHOULDER', s[2], s[3]) if (s[1] == 'OFF-SHOULDER' and col == 'NeckDesign') else s
    s = (s[0], s[1] + " LENGTH", s[2], s[3]) if (s[1] == 'SHORT' or s[1] == 'MEDIUM' or s[1] == 'KNEE' and col == 'Length') else s
    s = (s[0], 'LONG', s[2], s[3]) if (s[1] == 'MAXI' and col == 'Length') else s
    s = (s[0], s[1] + " COLLAR", s[2], s[3]) if ((s[1] == 'MAO' or s[1] == 'STAND UP' or s[1] == 'POLO') and col == 'CollarDesign') else s
    s = (s[0], s[1] + " FIT", s[2], s[3]) if ((s[1] == 'REGULAR' or s[1] == 'RELAXED' or s[1] == 'SLIM') and col == 'Fit') else s
    return s


def updateQuery(assosDF, engine):
    ###Create temp table and then update only the rows that belong to temp AND product###
    assosDF.to_sql("temp_table", schema ='dbo', con = engine, if_exists = 'replace', index = False)
    with engine.begin() as conn:
        conn.execute(constants.UPDATESQLQUERY)

if __name__ == "__main__":
#DATABASE/ASSOS#
    ###Read Table Products from SocialMedia DB###
    engine, query_db = connectDatabase()

    assosDF = pd.DataFrame(query_db)
    labelsDF = pd.DataFrame()
    
    ###Filter the old elements with the new one, Keep only the non-updated elements to improve efficiency, reset index due to slice###
    assosDF = assosDF[assosDF.loc[:,(constants.NRGATTRIBUTESID + constants.DEEPFASHIONATTRIBUTESID)].isnull().apply(lambda x: all(x), axis=1)]
    assosDF = assosDF.reset_index(drop=True)
    #print(assosDF.columns)
    #assosDF = assosDF.drop(constants.NRGATTRIBUTESID + constants.DEEPFASHIONATTRIBUTESID, axis = 1)
    labelsDF['ProductNo'] = assosDF['ProductNo'].copy()
    ###Metadata and Headline consists of information related to each row###

    metadata = assosDF['Metadata'].str.upper()
    headline = assosDF['SiteClothesHeadline'].str.upper()
#NRG#

    ###Read possible labels from Energiers###

    labelsNRG = pd.read_excel(constants.NRGATTRIBUTESPATH, sheet_name=constants.SHEETNAME)
    
    ###Create Variables with same name as the Energiers column names, to store the labels. Create new columns at assos with Energiers column names###

    for at in constants.NRGATTRIBUTES:
        locals()[str(at)] = labelsNRG[at].replace(' ', np.nan).dropna().unique()
        labelsDF[at] = np.empty((len(assosDF), 0)).tolist()
#DEEPFASHION#

    groups = preprocessDeepfashion()
    for attribute in constants.DEEPFASHIONATTRIBUTES:
        locals()[str(attribute)] = groups.get_group(str(attribute[0]) + str(attribute[1:]))
        labelsDF[attribute] = np.empty((len(assosDF), 0)).tolist()

#Preprocessing#

    ###Convert every label, metadata and headline to uppercase###
    for label in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        locals()[str(label)] = list(locals()[str(label)])
        locals()[str(label)] = [la.upper() for la in locals()[str(label)]]
    #DES LIGO TO TRAINING STO PYTORCH ME TO PREPROCESSING GIA METADATA KAI HEADLINE SAN TRAIN KAI TEST

    splitted_metadata = [s.split() if isinstance(s,str) else " " for s in metadata]
    splitted_headline = [s.split() if isinstance(s,str) else " " for s in headline]
    
    
    ###Search for occurences of labels in metadata and headline. For category length if the next word is not a kind of cat (Trousers etc) then it is propably wrong so we get rid of it.###
    cat = 'ProductCategory'
    for col in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        saved_meta = [(index, label, s.find(label), 1) for label in (locals()[str(col)]) for index,s in enumerate(metadata) if contains_word(s, label)]
        saved_head = [(index, label, s.find(label), 0) for label in (locals()[str(col)]) for index,s in enumerate(headline) if contains_word(s, label)]
        for s in (saved_meta + saved_head):
            s = editStrings(s)
            labelsDF.loc[s[0],col].append((s[1], s[2], s[3]))
            if col == 'Length':
                flag = 0
                for i,strSen in enumerate(splitted_metadata+splitted_headline):
                    for j,sen in enumerate(strSen[:-1]):
                        if sen == s[1] and 'SLEEVE' == strSen[j+1] and flag == 0:
                            flag = 1             
                if flag == 1:
                    #print(labelsDF.loc[s[0], col])
                    labelsDF.loc[s[0], col].remove((s[1], s[2], s[3]))
    
    ###Find similar words, for example -> rounded and round and one of them is discarded###

    for col in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        #Τhe next line sorts the elements of the respective columns alphabetically
        #labelsDF.loc[:, col] = pd.Series([sorted(list(ele)) for ele in labelsDF.loc[:, col]])
        #Τhe next line sorts the elements of the respective columns based on position and on metadata or headline (headline first and then position on each string)
        labelsDF.loc[:, col] = pd.Series([sorted(list(ele), key = lambda tup: (tup[2], tup[1])) for ele in labelsDF.loc[:, col]])
        labelsDF.loc[:, col] = pd.Series([list(map(lambda x: x[0], ele)) for ele in labelsDF.loc[:, col]])
        saved = defaultdict(list)
        for i, element in enumerate(labelsDF.loc[:, col]):
            if len(element) >= 2:
                for (k, l1),(ki, l2) in itertools.combinations(enumerate(element), 2): 
                    if similar(l1[0], l2[0]) >= 0.8:
                        saved[i].append((ki,l2))

        for key,value in saved.items():
            uni = np.unique(value, axis = 0)
            for index, x in uni:
                #We reverse because remove always pop outs the first element while we want the last
                labelsDF.loc[key, col].reverse()
                labelsDF.loc[key, col].remove(x)
                labelsDF.loc[key, col].reverse()
                
    ##Kane to sort se ola -> applymap
    ###Check if list is empty and in this case make it None###
    for col in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        for row in range(labelsDF.shape[0]):
            if not labelsDF.loc[row, col]:
                labelsDF.loc[row, col] = None
            else:
                labelsDF.loc[row, col] = ','.join(labelsDF.loc[row, col])
    ##MAYBE USE APPLYMAP for example
    #labelsDF.applymap(lambda x: None if not x else ','.join(x))
#Upload to SocialMedia DB#

    #Extract unique labels from metadata and headline
    labelsUnique = {col:list(labelsDF.loc[:, col].dropna().unique()) for col in labelsDF.loc[:, constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES].columns}

    #Read from database the labels 
    for name in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        locals()[str(name)+'_DB'] = pd.read_sql_query(constants.SELECTQUERY + str(name.upper()), engine)

    #Find which labels are already registered and register the unregistered ones, generate ID for the news
    for (key, value) in labelsUnique.items():
        locale=locals()
        temp = list(map(lambda x: x if x not in list((locale[str(key)+'_DB'].iloc[:, 1])) else None, value))
        keepNew = [i for i in temp if i]
        locale['new'+str(key)+'_DB'] = pd.DataFrame(columns = list(locale[str(key)+'_DB'].columns))
        for index, value in enumerate(keepNew):
            if len(value.split(',')) >= 2:
                keepValue = value.split(',')[0]
                genID = [locale[str(key)+'_DB'].iloc[index,0] for index, value in enumerate(locale[str(key)+'_DB'].iloc[:,1]) if keepValue == value]
            locale[str(key)+'_DB'].loc[locale[str(key)+'_DB'].shape[0]] = [locale[str(key)+'_DB'].shape[0]] + [value] + genID

            locale['new'+str(key)+'_DB'].loc[index] = [locale[str(key)+'_DB'].shape[0] - 1] + [value] + genID
        testing = labelsDF.merge(locale[str(key)+'_DB'], how = 'left', on=str(key))
        #https://stackoverflow.com/questions/14940743/selecting-excluding-sets-of-columns-in-pandas
        #df[df.columns[~df.columns.isin(['C','D'])]]
        temp = assosDF[assosDF.columns[~assosDF.columns.isin([str(key)+'ID'])]].merge(testing[['ProductNo', str(key)+'ID']], how = 'left', on = 'ProductNo')
        assosDF.loc[:,str(key)+'ID'] = temp.loc[:,str(key) + 'ID']

#Filter similar labels to keep the originals
    '''
    
    testDF = assosDF.iloc[:,1:]
    format = lambda x: max(similar(x, fromLabel)) for fromLabel in locals()['NeckDesign']
    #testDF.NeckDesign.map(lambda x,y: max(similar(x,y)) )
    map(lambda x,y: max(similar(x,y)),testDF.NeckDesign)
'''
    for name in (constants.NRGATTRIBUTES + constants.DEEPFASHIONATTRIBUTES):
        nameUpper = name.upper()
        locale['new'+str(name)+'_DB'].to_sql(nameUpper, schema ='dbo', con = engine, if_exists = 'append', index = False)

    updateQuery(assosDF, engine)
    

    print("--- %s seconds ---" % (time.time() - start_time))















