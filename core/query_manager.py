import core.config as config


# --------------------------------
#          S4F QueryManager
# --------------------------------
class QueryManager():
    def __init__(self, user=config.DEFAULT_USER):
        self.user = user
        self.dbName = config.DB_NAME
        self.engine = config.ENGINE


    def runInsertQuery(self, params=dict):
        """
        Execute "INSERT INTO" queries using a parameter dictionary.
        The dictionary should contain the table information as well as the fields
        and assigned values as key-value pairs, e.g

        param = {"table": "S4F.dbo.Product", "CreatedBy": "user", "UpdatedBy": "user"}
        """
        if "table" not in params.keys():
            raise ValueError('The "table" information is missing from "params" dictionary')
        else:
            params = self.parseParams(params)            
            
            fields = ', '.join(list(params.keys())[1:])
            values = ', '.join(list(params.values())[1:])
            query = "INSERT INTO %s.dbo.%s (%s)" % (self.dbName, params['table'], fields) + \
                " VALUES (%s)" % values
            self.runSimpleQuery(query)

    def parseParams(self, params):
        _params = params.copy()
        for k,v in _params.items():
            if k != 'table':
                # parse integers
                if type(v) == int:
                    params[k] = self.parseInt(v)
                # parse strings
                if type(v) == str:
                    params[k] = self.parseStr(v)
        
        # Add user information
        if self.user != config.DEFAULT_USER:
            params['CreatedBy'] = "'%s'" % self.user
            params['UpdatedBy'] = "'%s'" % self.user
        else:
            params['CreatedBy'] = self.user
            params['UpdatedBy'] = self.user
        return params

    def parseInt(self, i):
        return 'CAST(%s AS INTEGER)' % i
    
    def parseStr(self, i):
        return 'STRING_ESCAPE(\'%s\', \'json\')' % i



    def runSimpleQuery(self, query, args=None):
        """
        Execute input query, e.g.

        query = "SELECT * FROM S4F.dbo.Product"
        runQuery(query)

        Function argument "args" is used to handle "INSERT INTO" statements, e.g.

        query = "INSERT INTO %s.dbo.Log (LogType, Note, Details)" % self.dbName \
                    + " VALUES (CAST(%s AS INTEGER), %s, STRING_ESCAPE(%s, 'json'))"
        args=  logType, note, str(details)
        self.runQuery(query, args)
        """
        with self.engine.begin() as conn:
            if args:
                conn.execute(query, args)
            else:
                conn.execute(query)

