import sys
import logging
import os
#########################
# Globbal Logger        #
#########################
class GLogger:
    def __init__(self,name,
                 filename=None,
                 level=logging.INFO,
                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                 ):
        self.name = name
        self.filename = filename
        self.level = level
        self.format = format

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        self.formatter_console = logging.Formatter(self.format, datefmt='%m/%d/%Y %H:%M:%S')
        ch.setFormatter(self.formatter_console)
        self.logger.addHandler(ch)

        if self.filename:
            self.add_log_file(filename)
        self.logger.propagate = False

    def add_log_file(self,filename,level=None):
        fh = logging.FileHandler(filename)
        file_level = level or self.level
        fh.setLevel(file_level)
        fh.setFormatter(self.formatter_console)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def create_logger(logger_name,exp_params):
    log_file_name=os.path.join(exp_params.output_root_dir,exp_params.name+'.log')
    log_level=exp_params.log_level
    log_format=exp_params.log_format
    logger = GLogger(logger_name,filename=log_file_name,
                      level=log_level,format=log_format).get_logger()

    # if stdout_to_log:
    #     sys.stdout = StreamToLogger(logger, logging.INFO)
    #     sys.stderr = StreamToLogger(logger, logging.ERROR)
    return logger


class ScopedLogSplit(object):
    def __init__(self, logger):
        """
        Class for using context manager while logging

        usage:
        with ScopedLogSplit():
            {code}

        :param folder: (str) the logging folder
        :param format_strs: ([str]) the list of output logging format
        """
        self.logger = logger
        self.prev_stdout = None
        self.prev_stderr = None

    def __enter__(self):
        self.prev_stdout = sys.stdout
        self.prev_stderr = sys.stderr
        sys.stdout = StreamToLogger(self.logger, logging.INFO)
        sys.stderr = StreamToLogger(self.logger, logging.ERROR)

    def __exit__(self, *args):
        sys.stdout = self.prev_stdout
        sys.stderr = self.prev_stderr


def title(msg,n,ch='='):
    return "\n\n"+ch*n+" "+msg+" "+ch*n


global glogger

