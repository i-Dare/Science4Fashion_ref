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
from core.query_manager import QueryManager

PRODUCT_ATTRIBUTES = config.PRODUCT_ATTRIBUTES + ['Gender']

class MetadataAnnotator():
    '''
    Rensponsible for text annotation
    '''
    def __init__(self, user, *oids, loglevel=config.DEFAULT_LOGGING_LEVEL):
        self.user = user
        self.oids = oids
        self.loglevel = loglevel
        self.logging = S4F_Logger('TextAnnotationLogger', user=user, level=self.loglevel)
        self.logger = self.logging.logger
        self.helper = Helper(self.logging)
        self.db_manager = QueryManager(user=self.user)        

        ### Select Products to execute the text based annotation
        ## Prepare query
        # Merge product and color information from database
        attJoin, attSelect = '', []
        for i, attr in enumerate(PRODUCT_ATTRIBUTES):
            attSelect += ['attr%s.Description AS %s' % (i, attr)]
            attJoin += ' LEFT JOIN %s.dbo.%s AS attr%s\nON PRD.%s = attr%s.Oid ' \
                % (config.DB_NAME, attr, i, attr, i) 

        if len(oids) > 0:
            # Select from provided product IDs
            where = ' OR '.join(['PRD.Oid=%s' % i for i in  oids])            
        else:
            # Select from products with missing attributes
            where = ' OR '.join(['PRD.%s is NULL' % attr for attr in  PRODUCT_ATTRIBUTES])

        query = '''SELECT PRD.Oid, PRD.Description, PRD.Metadata, PRD.Composition, PRD.ForeignComposition,
                    C.Label, C.LabelDetailed, C.Red, C.Blue, C.Green,
                    PC.Ranking,
                    %s
                FROM %s.dbo.Product AS PRD 
                LEFT JOIN %s.dbo.ProductColor AS PC
                ON PC.Product=PRD.Oid
                LEFT JOIN %s.dbo.ColorRGB AS C
                ON PC.ColorRGB = C.Oid
                %s
                WHERE %s''' % (','.join(attSelect), config.DB_NAME, config.DB_NAME, config.DB_NAME, 
                attJoin, where)
        
        products_df = self.db_manager.runSimpleQuery(query, get_identity=True)
        products_df = products_df.drop_duplicates(ignore_index=True)
        self.products_df = self.product_preprocessing(products_df)


    def product_preprocessing(self, products_df):
        ## Merging results
        # 
        start_time = time.time() 
        attributes_df = products_df.groupby('Oid').first()[PRODUCT_ATTRIBUTES + 
                ['Metadata', 'Description', 'Composition', 'ForeignComposition']]
        # Color data merging
        grouped_products_df = (products_df.groupby('Oid')
                .apply( lambda row:  list(row['Label'] ))
                .str.join(',')
                .str.split(',', expand=True))
        attributes_df.loc[:, ['ColorRanking%s' % n for n in grouped_products_df.columns]] = grouped_products_df.values
        # Text merging
        text_columns = config.PRODUCT_ATTRIBUTES + ['Label', 'LabelDetailed', 'Composition', 'ForeignComposition']
        attributes_df['extended_metadata'] = (products_df.groupby('Oid')
                .apply( lambda row:  [list(set(row[col])) for col in text_columns ] )
                .apply( lambda row:  [r.replace(' ', '_') for r in (sum(row, []))  if not pd.isna(r)] )
                .str.join(' ')).values
        # Text preprocessing
        attributes_df.loc[:, 'Metadata'] = (attributes_df.loc[:, 'Metadata'].fillna('')
                .apply(self.helper.preprocess_metadata))
        attributes_df.loc[:, 'processed_extended_metadata'] = (attributes_df.loc[:, 'extended_metadata']
                .apply(self.helper.preprocess_metadata))
        # # Filter out tokens already in the Metadata column
        # attributes_df.loc[:, 'processed_extended_metadata'] = (attributes_df.loc[:, ['Metadata', 'processed_extended_metadata']]
        #       .apply(lambda row: [r  for r in row['processed_extended_metadata'].split() if r not in row['Metadata']] , axis=1)
        #       .str.join(' '))
        # Final merge to 'Metadata' columns selecting only unique tokens
        attributes_df['Metadata'] = (attributes_df['Metadata']
                .str.cat(attributes_df['processed_extended_metadata']
                .astype(str), sep=' ').apply(lambda x: ' '.join(set(x.split())) ))
        self.logger.debug('Processed %s records in %s seconds' % (len(attributes_df), round(time.time() - start_time, 2)))
        return attributes_df.loc[:, PRODUCT_ATTRIBUTES+ ['Description', 'Metadata']].reset_index()

    def execute_annotation(self,):        
        try:
            start_time = time.time()
            self.logger.info("Executing text annotation for %s unlabeled products" % len(self.products_df))
            if not self.products_df.empty:
                labels_df = pd.DataFrame()
                labels_df['Oid'] = self.products_df['Oid'].copy()
                # Metadata and Headline consists of information related to each row 
                metadata = self.products_df['Metadata'].str.lower()
                headline = self.products_df['Description'].str.lower()

                # Create a dictionary of dataframes to read product attribute data from the database
                dfDict = {}
                # Create a dictionary of variables with same name as the product attribute names, 
                # to store the labels. 
                attrDict = {}                
                for attr in PRODUCT_ATTRIBUTES:
                    # Populate dataframe dictionary
                    dfDict[str(attr)+'_DB'] = self.db_manager.runSelectQuery(params={'table': attr})
                    # Populate product attribute dictionary
                    attrDict[str(attr)] = dfDict[str(attr)+'_DB']['Description'].str.lower().tolist()
                    labels_df[attr] = np.empty((len(self.products_df), 0)).tolist()
               
                # Preprocessing 
                # Convert every label, metadata and headline to uppercase 
                splitted_metadata = [s.split() if isinstance(s, str) else " " for s in metadata]
                splitted_headline = [s.split() if isinstance(s, str) else " " for s in headline]    
                
                # Search for occurences of labels in metadata and headline. For attribute LENGTH if the next 
                # word is not a kind of category (Trousers etc) then it is propably wrong so we get rid of it.
                # Store label accurences as tuple in the form of (index, label, position, 1|0)
                # index: the absolute position of the product in "labels_df" 
                # label: a specific value of product a attribute
                # position: the position of the "label" in the metadata or headline lists
                for attr in PRODUCT_ATTRIBUTES:
                    saved_meta = [(index, label, s.find(label), 1) for label in attrDict[str(attr)] 
                            for index,s in enumerate(metadata) if contains_word(s, label)]
                    saved_head = [(index, label, s.find(label), 0) for label in attrDict[str(attr)]
                            for index,s in enumerate(headline) if contains_word(s, label)]
                    for s in (saved_meta + saved_head):
                        s = editStrings(s, attr)
                        labels_df.loc[s[0],attr].append((s[1], s[2], s[3]))
                        if attr == 'Length':
                            flag = 0
                            for i, strSen in enumerate(splitted_metadata+splitted_headline):
                                for j, sen in enumerate(strSen[:-1]):
                                    if sen == s[1] and 'SLEEVE' == strSen[j+1] and flag == 0:
                                        flag = 1             
                            if flag == 1:
                                labels_df.loc[s[0], attr].remove((s[1], s[2], s[3]))
                
                # Find similar words, for example -> rounded and round and one of them is discarded 
                for attr in PRODUCT_ATTRIBUTES:
                    # Sort the elements of the respective columns based on position on metadata or headline (headline first and then position on each string)
                    labels_df.loc[:, attr] = pd.Series([sorted(list(ele), key = lambda tup: (tup[2], tup[1])) for ele in labels_df.loc[:, attr]])
                    labels_df.loc[:, attr] = pd.Series([list(map(lambda x: x[0], ele)) for ele in labels_df.loc[:, attr]])
                    saved = defaultdict(list)
                    for i, element in enumerate(labels_df.loc[:, attr]):
                        if len(element) >= 2:
                            for (k, l1),(ki, l2) in itertools.combinations(enumerate(element), 2): 
                                if similar(l1, l2) >= 0.8:
                                    saved[i].append((ki,l2))

                    for key,value in saved.items():
                        uni = np.unique(value, axis = 0)
                        for index, x in uni:
                            # We reverse because remove always pop outs the first element while we want the last
                            labels_df.loc[key, attr].reverse()
                            labels_df.loc[key, attr].remove(x)
                            labels_df.loc[key, attr].reverse()

                # Check if list is empty and in this case make it None 
                for attr in PRODUCT_ATTRIBUTES:
                    labels_df.loc[:, attr] = labels_df[attr].apply(lambda x: ','.join(x) if x else None)
                #Extract unique labels from metadata and headline
                labelsUnique = {attr:set([l for label in labels_df[attr].unique() 
                        if label for l in label.split(',')]) 
                        for attr in labels_df.loc[:, PRODUCT_ATTRIBUTES].columns}

                # Update the DB tables with the new attributes                
                for (table, values) in labelsUnique.items():
                    if table != 'Gender':
                        self.logger.info('Updating product attribute table %s' % table)
                        for v in values:
                            uniq_params = {'table': table, 'Description': v.lower()}
                            params = {'table': table, 'Description': v.lower(), 'AlternativeDescription': '', 
                                    'ProductCategory': None, 'Active': True, 'OptimisticLockField': None}
                            self.db_manager.runCriteriaInsertQuery(
                                                                uniq_params=uniq_params, 
                                                                params=params, 
                                                            )
                                
                self.logger.info('Update product attributes')
                ## Update Product table with the foreign key values of the updated attributes
                # re-load from database the updated attribute tables and create a dataframe for each 
                dfDict = {}
                for attr in PRODUCT_ATTRIBUTES:
                    dfDict[str(attr)+'_DB'] = self.db_manager.runSelectQuery(params={'table': attr})
                    for col in dfDict[str(attr)+'_DB'].columns:
                        try:                        
                            dfDict[str(attr)+'_DB'][col] = dfDict[str(attr)+'_DB'][col].str.lower()
                        except:
                            pass

                # Update Product table for each attribute
                for attr in PRODUCT_ATTRIBUTES:
                    # If there are multiple attributes, select the first
                    labels_df.loc[labels_df[attr].notnull(), attr] = (labels_df[labels_df[attr]
                                                                        .notnull()][attr]
                                                                        .apply(lambda x: x.split(',')[0] if x else None))
                    # Merge the updated attribute values to Product table
                    merged_df = labels_df.merge(dfDict[str(attr)+'_DB'], left_on=attr, right_on='Description')[['Oid_x', 'Oid_y']]
                    for i, row in merged_df.iterrows():                        
                        # If values is NA for product attribute, update with the captured value
                        if pd.isna(self.products_df.loc[self.products_df['Oid']==row['Oid_x'], attr].values[0]):
                            productID =  row['Oid_x']
                            attrID = row['Oid_y']
                            attrVal = dfDict[str(attr)+'_DB'].set_index('Oid').loc[attrID, 'Description']

                            self.logger.debug("Update %s attribute of %s with %s" % (attr,productID, attrVal), 
                                    extra={'Product': productID})
                            # Update record
                            uniq_params = {'table': 'Product', 'Oid': productID}
                            params = {'table': 'Product', attr: attrID}
                            self.db_manager.runCriteriaUpdateQuery(uniq_params=uniq_params, params=params)
                # Batch update Product Metadata
                self.logger.debug('Update product metadata')
                table = 'Product'
                columns = ['Oid', 'Metadata']
                self.db_manager.runBatchUpdate(table, self.products_df[columns], 'Oid')
               
            self.logger.info("--- Finished text annotation of %s records in %s seconds ---" % (len(self.products_df), 
                    round(time.time() - start_time, 2)))
        except Exception as ex:
            self.logger.warn_and_exit(ex)
        self.logger.close()


#Relevant similarity function
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def contains_word(s, w):
    return f' {w} ' in f' {s} '

def editStrings(s, attr):
    # Check for specific values of s and change the values to match those of Energiers 
    s = (s[0], 'V NECK', s[2], s[3]) if (s[1] == 'V-NECK' and attr == 'NeckDesign') else s
    s = (s[0], 'OFF SHOULDER', s[2], s[3]) if (s[1] == 'OFF-SHOULDER' and attr == 'NeckDesign') else s
    s = (s[0], s[1] + " LENGTH", s[2], s[3]) if (s[1] == 'SHORT' or s[1] == 'MEDIUM' or s[1] == 'KNEE' and attr == 'Length') else s
    s = (s[0], 'LONG', s[2], s[3]) if (s[1] == 'MAXI' and attr == 'Length') else s
    s = (s[0], s[1] + " COLLAR", s[2], s[3]) if ((s[1] == 'MAO' or s[1] == 'STAND UP' or s[1] == 'POLO') and attr == 'CollarDesign') else s
    s = (s[0], s[1] + " FIT", s[2], s[3]) if ((s[1] == 'REGULAR' or s[1] == 'RELAXED' or s[1] == 'SLIM') and attr == 'Fit') else s
    return s


if __name__ == "__main__":
    user = sys.argv[1]
    oids = sys.argv[2:-1]
    loglevel = sys.argv[-1]
    text_annotator = MetadataAnnotator(user, *oids, loglevel=loglevel)
    text_annotator.execute_annotation()