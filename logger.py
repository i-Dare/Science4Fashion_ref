from datetime import datetime
import logging
import os
import config
import sys
import traceback as tb

# import logging
# import threading

class MyLogger(logging.Logger):

    def __init__(self, name, level = logging.NOTSET):
        return super(MyLogger, self).__init__(name, level)        

    def warn_and_exit(self, ex, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.WARNING):
            msg = ''.join(tb.format_exception(None, ex, ex.__traceback__))
            self._log(logging.WARNING, msg, args, **kwargs)
            sys.exit(1)        

    def warn_and_trace(self, ex, *args, **kwargs):
        """
        Log 'msg % args' with severity 'WARNING'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.warning("Houston, we have a %s", "bit of a problem", exc_info=1)
        """
        if self.isEnabledFor(logging.WARNING):
            msg = ''.join(tb.format_exception(None, ex, ex.__traceback__))
            self._log(logging.WARNING, msg, args, **kwargs)
            sys.exit(1)



class S4F_Logger():
    def __init__(self, name, level=logging.DEBUG, logfile=None, user=config.DEFAULT_USER):
        self.level = level
        self.name = name
        self.logdir = config.LOGDIR

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.logger = self.initLogger(name, logfile,  user)

    def initLogger(self, name, logfile=None,  user=config.DEFAULT_USER):
        os.environ['PYTHONUNBUFFERED'] = "1"
        logging.setLoggerClass(MyLogger)
        # Setup formatter
        formatter = logging.Formatter('[%(asctime)s]  %(levelname)-2s::  %(message)-5s (%(name)s)')
        
        # Get or create a logger
        logger = logging.getLogger(name)  
        
        # Set level
        logger.setLevel(self.level)
        
        # define file and console handler and set formatter
        stdHandler = logging.StreamHandler()
        stdHandler.setFormatter(formatter)
        if logfile:
            fileHandler = logging.FileHandler(os.path.join(self.logdir, logfile))
        else:
            now = datetime.now().strftime('%Y-%m-%d')
            logfile = 'tmp_%s.log' % now
            fileHandler = logging.FileHandler(os.path.join(self.logdir, logfile))
        fileHandler.setFormatter(formatter)

        # add file handlers to logger
        logger.addHandler(stdHandler)
        logger.addHandler(fileHandler)

        logger.info('Start logging')
        return logger
            

        def initLogger2(self, loggerName, logfile=None,  user=config.DEFAULT_USER):
            os.environ['PYTHONUNBUFFERED'] = "1"

            # Setup formatter
            formatter = logging.Formatter('[%(asctime)s]  %(levelname)-2s::  %(message)-5s (%(name)s)')
            
            # Get or create a logger
            self.logger = logging.getLogger(loggerName)  
            
            # Set level
            self.logger.setLevel(self.level)

            # define file and console handler and set formatter
            stdHandler = logging.StreamHandler()
            stdHandler.setFormatter(formatter)
            if logfile:
                fileHandler = logging.FileHandler(os.path.join(self.logdir, logfile))
            else:
                now = datetime.now().strftime('%Y-%m-%d')
                logfile = 'tmp_%s.log' % now
                fileHandler = logging.FileHandler(os.path.join(self.logdir, logfile))
            fileHandler.setFormatter(formatter)

            # add file handlers to logger
            self.logger.addHandler(stdHandler)
            self.logger.addHandler(fileHandler)

            self.logger.info('Start logging')
            return self.logger
            

    # def setLogger(self,):
    #    pass





class SqlHandler(logging.StreamHandler):
    """
    A handler class which writes formatted logging records sql database.
    """
    def __init__(self, filename, connection, mode='a', encoding=None, delay=False):
        """
        Open the specified file and use it as the stream for logging.
        """
        # Issue #27493: add support for Path objects to be passed in
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        return open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        if self.stream is None:
            self.stream = self._open()
        logging.StreamHandler.emit(self, record)

    def __repr__(self):
        level = logging.getLevelName(self.level)
        return '<%s %s (%s)>' % (self.__class__.__name__, self.baseFilename, level)
   