import numpy as np
import pandas as pd
import re

import core.config as config



def returnDataFrame(df):
    for col in df.columns:
        # parse numeric
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def filtering(filters):
    selection = '*'
    if filters and type(filters)==list:
        selection = ','.join(['t.%s' % col for col in filters])
    if type(filters)==str and 'TOP'.lower() in filters.lower():
        selection = filters
    return selection

# --------------------------------
#          S4F QueryManager
# --------------------------------
class QueryManager():
    def __init__(self, user=config.DEFAULT_USER):
        self.user = user
        self.dbName = config.DB_NAME
        self.engine = config.ENGINE

    def runInsertQuery(self, params=dict, get_identity=False):
        """
        Execute "INSERT INTO" queries using a parameter dictionary.
        The 'params' dictionary should contain the table information as well as the fields
        and assigned values as key-value pairs, e.g

        param = {"table": "Brand", "Description": "Adidas"}
        """
        if "table" not in params.keys():
            raise ValueError('The "table" information is missing from "params" dictionary')
        else:
            table = params['table']
            # parse parameters
            params = self.parseParams(params, has_owner=True)            
            
            # prepare INSERT INTO statement
            fields = ', '.join([k for k,v in params.items() if  type(v) is not bytes])
            values = ', '.join([v for k,v in params.items() if  type(v) is not bytes])

            bins = []
            for k,v in params.items():
                if type(v) is bytes:
                    fields += ', %s' % k
                    values += ', %s'
                    bins.append(v)
       
            query = "INSERT INTO %s.dbo.%s (%s)" % (self.dbName, table, fields) + \
                " VALUES (%s)" % values
            return self.runSimpleQuery(query, args=tuple(bins,), get_identity=get_identity)

    def runCriteriaInsertQuery(self, uniq_params=dict, params=dict, get_identity=False):
        """
        Execute "IF NOT EXISTS ... INSERT INTO ..." queries using a unique parameter dictionary 
        for the constraining parameters and parameter dictionary for the input parameters.
        The 'uniq_params' dictionary should contain the fields and the values of criteria. 
        The 'params' dictionary should contain the table information as well as the fields
        and assigned values as key-value pairs, e.g

        uniq_params = {"Description": "Adidas"}
        param = {"table": "Brand", "Description": "Adidas"}
        """
        if "table" not in params.keys():
            raise ValueError('The "table" information is missing from "params" dictionary')
        else:
            # prepare criteria statement
            df = self.runSelectQuery(uniq_params)
            if df.empty:
                # If selection is empty, proceed to add new information
                return self.runInsertQuery(params=params, get_identity=get_identity)
            else:
                return df
                
    def runCriteriaUpdateQuery(self, uniq_params=dict, params=dict, get_identity=False):
        """
        Execute "UPDATE ... SET ... WHERE ...." queries using a unique parameter dictionary 
        for the constraining parameters and parameter dictionary for the input parameters.
        The 'uniq_params' dictionary should contain the fields and the values of criteria. 
        The 'params' dictionary should contain the table information as well as the fields
        and assigned values as key-value pairs, e.g

        uniq_params = {"table": "Brand", "Description": "Adidas"}
        param = {"table": "Brand", "Description": "Adidas"}
        """
        if "table" not in params.keys():
            raise ValueError('The "table" information is missing from "params" dictionary')
        else:
            table = params['table']

            # prepare criteria statement
            df = self.runSelectQuery(uniq_params)
            if not df.empty:
                # parse parameters
                params = self.parseParams(params, has_owner=True)  
            
                # prepare query statement
                values = ', '.join(['%s = %s' % (k,v) for k,v in params.items() if k!='table'])
                values += ', UpdatedOn = GETUTCDATE()'
                where = " Oid=%s" % df.loc[0, 'Oid']
                query = "UPDATE %s.dbo.%s SET %s WHERE %s" % (self.dbName, table, values, where)
                self.runSimpleQuery(query, get_identity=get_identity)
                return df

    def runBatchUpdate(self, table, df, criteria_col, step=config.BATCH_STEP):
        """
        Execute "UPDATE table... SET column1 = CASE column2... WHERE ...." queries using the argument 
        "criteria_col" to drive the case by case update.
        The "df" argument must be a DataFrame containing the "Oid" column as an identifier of the 
        record to update and the columns for update, e.g
        
        df = pd.DataFrame({'Oid': [1, 2, 3], 'Brand': ['Adidas', 'Nike', 'ALTRA']})
        criteria_col = 'Oid'
        table = 'Product'
        """
        df = df.reset_index().drop(columns=['index'])
        for i in df.index[::step]:
            for col in set(df.columns) - set([criteria_col]):
                chunk = df.loc[df.index[i:i+step], [criteria_col, col]]
                when = ['WHEN %s THEN %s' % 
                        (row[criteria_col], self.parseItem(row[col])) for _, row in chunk.iterrows() 
                        if self.parseItem(row[col])!='NULL']
                where = ', '.join(map(str, chunk[criteria_col].values.tolist()))
                if len(when)>0:
                    query = """UPDATE %s.dbo.%s 
                            SET %s = CASE %s
                            %s
                            END
                            WHERE %s IN (%s)""" % (config.DB_NAME, table, col, criteria_col, 
                                                   ' \n '.join(when), criteria_col, where)
                    self.runSimpleQuery(query)        
            

    def runSelectQuery(self, params=dict, filters=None):
        """
        Execute "SELECT" queries using a parameter dictionary.
        The 'params' dictionary should contain the table information as well as the fields
        and assigned values as key-value pairs, e.g

        param = {"table": "Brand", "Description": "Adidas"}
        """
        if "table" not in params.keys():
            raise ValueError('The "table" information is missing from "params" dictionary')
        else:
            table = params['table']
            params = self.parseParams(params)    
            if len(params.keys())>0:
                # prepare criteria statement
                where = ''
                for k,v in params.items():
                    if k!='table':
                        if where == '' and v == 'NULL':
                            where = 't.%s is NULL' % k
                        elif where == '' and v != 'NULL':
                            where = 't.%s = %s' % (k,v)
                        elif where != '' and v == 'NULL':
                            where += ' AND t.%s is NULL' % k
                        else:
                            where += ' AND t.%s = %s' % (k,v)

                query = "SELECT %s FROM  %s.dbo.%s t WHERE %s" % (filtering(filters), self.dbName, 
                        table, where)
            else:
                query = "SELECT %s FROM  %s.dbo.%s t" % (filtering(filters), self.dbName, table)
            return self.runSimpleQuery(query, get_identity=True)

    def parseItem(self, item):
        """
        Parses single object, used in batch commit processes
        """
        # parse floats
        if 'float' in str(type(item)).lower():
            if np.isnan(item):
                return 'NULL'
            else:
                return self.parseFloat(item)
        # parse integers
        if 'int' in str(type(item)).lower():
            return self.parseInt(item)
        # parse strings
        if type(item) == str:
            return self.parseStr(item)
        # parse booleans
        if type(item) == bool:
            return self.parseBool(item)
        # parser decimal
        if 'decimal' in str(type(item)):
            return self.parseDecimal(item)
        # ignore None values and 'table' key
        if item is None:
            return 'NULL'
                

    def parseParams(self, params, has_owner=False):
        """
        Parses query parameters and adds 'user' information if 'has_owner' is True
        """
        _params = params.copy()
        for k,v in params.items():
            # parse floats
            if 'float' in str(type(v)).lower():
                if np.isnan(v):
                    del _params[k]
                    continue
                else:
                    _params[k] = self.parseFloat(v)
            # parse integers
            if 'int' in str(type(v)).lower():
                _params[k] = self.parseInt(v)
            # parse strings
            if type(v) == str:
                _params[k] = self.parseStr(v)
            # parse booleans
            if type(v) == bool:
                _params[k] = self.parseBool(v)
            # parse decimal
            if 'decimal' in str(type(v)):
                _params[k] = self.parseDecimal(v)
            # parse timestamps
            if 'timestamp' in str(type(v)).lower():
                _params[k] = self.parseStr(str(v))
            # parse UUID
            if 'uuid' in str(type(v)).lower():
                _params[k] = self.parseStr(str(v))
            # ignore None values and 'table' key
            if v is None or k == 'table':
                del _params[k]
                continue
        
        # Add user information if owner is needed and is not already set beforehand as parameter
        if has_owner and 'CreatedBy' not in _params.keys() and 'UpdatedBy' not in _params.keys():
            if self.user != config.DEFAULT_USER:
                _params['CreatedBy'] = "'%s'" % self.user
                _params['UpdatedBy'] = "'%s'" % self.user
            else:
                _params['CreatedBy'] = self.user
                _params['UpdatedBy'] = self.user
        return _params

    def parseFloat(self, i):
        return 'CAST(%s AS FLOAT)' % i

    def parseInt(self, i):
        return 'CAST(%s AS INTEGER)' % i
    
    def parseStr(self, i):
        if i == 'NULL':
            return i
        elif i.startswith('http'):
            return "'%s'" % i
        elif re.search(r'(http(?=s|:)\S+)', i):
            return "'%s'" % i
        else:
            return "STRING_ESCAPE('%s', \'json\')" % i.replace("'", "''")#.replace('%', '%%')

    def parseBool(self, i):
        return str(1) if i else str(0)

    def parseDecimal(self, i):
        return self.parseFloat(float(i))

    def runSimpleQuery(self, query, args=None, get_identity=False, filters=None):
        """
        Execute input query. Set 'get_identity=True' to  return the results in a DataFrame , e.g.

        query = "SELECT * FROM S4F.dbo.Product"
        df = runSimpleQuery(query) # Returns empty DataFrame

        or

        query = "SELECT * FROM S4F.dbo.Product"
        df = runSimpleQuery(query, get_identity=True) 
        """
        with self.engine.begin() as conn:
            if get_identity:
                ## Fetching query rows
                # parse table name from query
                table = re.findall(r'\b(FROM|INTO|UPDATE)\b\s(.+?)\s', query)[0][1]
                    
                query = query + '\n SELECT %s FROM %s t WHERE t.Oid = SCOPE_IDENTITY()' % \
                        (filtering(filters), table)
                if args:
                    try:
                        result = conn.execute(query, args)
                    except Exception as e: 
                        return e                        
                else:                    
                    try:
                        result = conn.execute(query)
                    except Exception as e: 
                        return e
                records  = result.fetchall()
                ## Create DataFrame from returned queries
                # Set DataFrame rows
                if records:
                    rows = [r.values() for r in records]
                    df = pd.DataFrame(columns=records[0].keys(), data=rows)
                else:
                    df = pd.DataFrame() 
                return returnDataFrame(df)
            else:
                # Execute "silent" query
                conn.execute(query)

    ## Returns the last record ID from the specified column and table
    #
    def getLastRecordID(self, table, where=None, filters=None):
        if where:
            query = "SELECT TOP 1 %s FROM %s.dbo.%s t  %s  ORDER BY Oid DESC" % (filtering(filters), 
                    config.DB_NAME, table, where)
        else:
            query = "SELECT TOP 1 %s FROM %s.dbo.%s t ORDER BY Oid DESC" % (filtering(filters), 
                    config.DB_NAME, table)
        with self.engine.begin() as conn:    
            result = conn.execute(query)
            oid = result.fetchone()
        return oid[0] if oid else None      
