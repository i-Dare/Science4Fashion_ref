import numpy as np
import pandas as pd
import re

import core.config as config


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
                setValues = ', '.join(['%s = %s' % (k,v) for k,v in params.items() if k!='table'])
                setValues += ', UpdatedOn = GETDATE()'
                where = " Oid=%s" % df.loc[0, 'Oid']
                query = "UPDATE %s.dbo.%s SET %s WHERE %s" % (self.dbName, table, setValues, where)
                self.runSimpleQuery(query, get_identity=get_identity)
                return df
            

    def runSelectQuery(self, params=dict):
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
                where = ' AND '.join(['t.%s = %s' % (k,v) for k,v in params.items() if k!='table'])
                query = "SELECT * FROM  %s.dbo.%s t WHERE %s" % (self.dbName, table, where)
            else:
                query = "SELECT * FROM  %s.dbo.%s t" % (self.dbName, table)
            return self.runSimpleQuery(query, get_identity=True)

    def parseParams(self, params, has_owner=False):
        """
        Parses query parameters and adds 'user' information if 'has_owner' is True
        """
        _params = params.copy()
        for k,v in _params.items():
            # parse floats
            if 'float' in str(type(v)).lower():
                params[k] = self.parseFloat(v)
            # parse integers
            if 'int' in str(type(v)).lower():
                params[k] = self.parseInt(v)
            # parse strings
            if type(v) == str:
                params[k] = self.parseStr(v)
            # parse booleans
            if type(v) == bool:
                params[k] = self.parseBool(v)
            # parser decimal
            if 'decimal' in str(type(params[k])):
                params[k] = self.parseDecimal(v)
            # ignore None values and 'table' key
            if v is None or k == 'table':
                del params[k]
        
        # Add user information if owner is needed and is not already set beforehand as parameter
        if has_owner and 'CreatedBy' not in params.keys() and 'UpdatedBy' not in params.keys():
            if self.user != config.DEFAULT_USER:
                params['CreatedBy'] = "'%s'" % self.user
                params['UpdatedBy'] = "'%s'" % self.user
            else:
                params['CreatedBy'] = self.user
                params['UpdatedBy'] = self.user
        return params

    def parseFloat(self, i):
        return 'CAST(%s AS FLOAT)' % i

    def parseInt(self, i):
        return 'CAST(%s AS INTEGER)' % i
    
    def parseStr(self, i):
        return 'STRING_ESCAPE(\'%s\', \'json\')' % i

    def parseBool(self, i):
        return str(1) if i else str(0)

    def parseDecimal(self, i):
        return self.parseFloat(float(i))


    def runSimpleQuery(self, query, args=None, get_identity=False):
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
                    
                query = query + '\n SELECT * FROM %s t WHERE t.Oid = SCOPE_IDENTITY()' % table
                if args:
                    result = conn.execute(query, args)
                else:
                    result = conn.execute(query)
                records  = result.fetchall()
                ## Create DataFrame from returned queries
                # Set DataFrame rows
                if records:
                    rows = np.asarray([r.values() for r in records])
                    df = pd.DataFrame(columns=records[0].keys(), data=rows)
                else:
                    df = pd.DataFrame() 
                return df
            else:
                # Execute "silent" query
                conn.execute(query)

    ## Returns the last record ID from the specified column and table
    #
    def getLastRecordID(self, table, where=None):
        if where:
            query = "SELECT TOP 1 * FROM %s.dbo.%s t  %s  ORDER BY Oid DESC" % (config.DB_NAME, table, where)
        else:
            query = "SELECT TOP 1 * FROM %s.dbo.%s t ORDER BY Oid DESC" % (config.DB_NAME, table)
        with self.engine.begin() as conn:    
            result = conn.execute(query)
            oid = result.fetchone()
        return oid[0] if oid else None            

