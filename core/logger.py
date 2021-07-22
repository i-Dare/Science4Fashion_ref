from datetime import datetime
import logging
import os
import core.config as config
from core.query_manager import QueryManager
import sys
import traceback as tb


_NOTSET, _INFO, _COMPLETE, _SKIPPED, _WARNING, _ERROR = -1, 0, 1, 2, 3, 4
LEVEL_DICT = {
                "INFO": _INFO,
                "COMPLETE": _COMPLETE,
                "SKIPPED": _SKIPPED,
                "WARNING": _WARNING,
                "ERROR": _ERROR,
                }
logging.addLevelName(_COMPLETE, "COMPLETE")                 
logging.addLevelName(_SKIPPED, "SKIPPED")     



# --------------------------------
#          S4F Logger
# --------------------------------
class S4F_Logger():
    def __init__(self, name, level=1, user=config.DEFAULT_USER):
        self.level = level
        self.name = name    
        self.user = user
        self.logger = self.initLogger(name, user)

    def initLogger(self, name, user=None):
        os.environ['PYTHONUNBUFFERED'] = "1"
        logging.setLoggerClass(CustomLogger)
        # Setup formatter
        if user:
            self.user = user
  
        formatter = MyFormatter(fmt='[%(asctime)s]  %(levelname)-2s:%(user)s:  %(message)-5s \
                (%(name)s)', user=self.user)
        
        # Get or create a logger
        logger = logging.getLogger(name)  
        
        # Set level
        logger.setLevel(self.level)
        
        # define console and sql handler and set formatter
        stdHandler = logging.StreamHandler()
        stdHandler.setFormatter(formatter)

        sqlHandler = SqlHandler()
        sqlHandler.setFormatter(formatter)

        # add file handlers to logger
        logger.addHandler(stdHandler)
        logger.addHandler(sqlHandler)

        if self.user is config.DEFAULT_USER:
            logger.warning('Start logging for user %s' % self.user) 
        else:
            logger.info('Start logging for user %s'  % self.user)
        return logger       


# --------------------------------
#       Main Logger
# --------------------------------
class CustomLogger(logging.Logger):

    def warn_and_exit(self, ex: Exception, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING', return code 1 and print the traceback.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warn_and_exit("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.WARNING):
            msg = ''.join(tb.format_exception(None, ex, ex.__traceback__))
            self._log(logging.WARNING, msg, args, **kwargs)
            sys.exit(1)        

    def warn_and_trace(self, ex: Exception, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING' and print the traceback.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.WARNING):
            msg = ''.join(tb.format_exception(None, ex, ex.__traceback__))
            self._log(logging.WARNING, msg, args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
            """
            Log 'msg % args' with severity 'ERROR'.

            To pass exception information, use the keyword argument exc_info with
            a true value, e.g.

            logger.error("Houston, we have a %s", "major problem", exc_info=1)
            """
            if self.isEnabledFor(logging.ERROR):
                self._log(logging.ERROR, msg, args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'INFO'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.info("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def skipped(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'SKIPPED'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.skipped("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(_SKIPPED):
            self._log(_SKIPPED, msg, args, **kwargs)

    def complete(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'COMPLETE'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.complete("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(_COMPLETE):
            self._log(_COMPLETE, msg, args, **kwargs)
    
    def close(self,):
        handlers = self.handlers.copy()
        for l in handlers:
            self.removeHandler(l)
            l.flush()
            l.close()

# --------------------------------
#       Logging Formatter
# --------------------------------
class MyFormatter(logging.Formatter):
        
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
    
    #
    # Extend the native "format" function to add user information in log message and change level ID 
    # 
    def format(self, record):
        record.user = self.user
        record.levelno = LEVEL_DICT[record.levelname]
        return super().format(record)


# --------------------------------
#          SQL Handler
# --------------------------------
class SqlHandler(logging.Handler):
    """
    A handler class which writes straight to the S4F databaset Log table
    """

    def __init__(self, level=logging.NOTSET):
        """
        Get DB information
        """        
        self.terminator = ';'
        super().__init__(level=level)


    # Extend the native "emit" function to write logs to S4F DB
    # 
    def emit(self, record):
        try:
            
            message = self.format(record)
            user = record.user
            name = record.name
            level = record.levelname
            logType = record.levelno
            note = "%s:%s" % (name, level)
            details = record.msg
            
            params = {'table': 'Log', 'LogType': logType, 'Note': note, 'Details': str(details)}
            db_manager = QueryManager(user=user)
            db_manager.runInsertQuery(params)
            # flush
            self.flush()

        except Exception as ex:
            msg = ''.join(tb.format_exception(None, ex, ex.__traceback__))
            # record.msg = '%s  Traceback: %s' % (record.msg, msg)
            self.handleError(record)

