# Code from https://stackoverflow.com/a/8349076
import logging
import sys

from utils.colors import Colors
from utils.constants import ROOT_PATH


class ColorLogger(logging.Formatter):
    """Class for colored python logs for the application.

    Use only if terminal supports coloring.
    """

    prefix = '{}[%(name)s]{} - '.format(Colors.CVIOLET2, Colors.ENDC)
    err_fmt = prefix + "{}%(levelname)s (%(module)s: %(lineno)d): {}{}%(message)s{}".format(
                                                                Colors.FAIL, Colors.ENDC, Colors.CBOLD, Colors.ENDC)
    info_fmt = prefix + "{}%(levelname)s: {}{}%(message)s{}".format(
                                                                Colors.CBLUE2, Colors.ENDC, Colors.CBOLD, Colors.ENDC)
    wrn_fmt = prefix + "{}%(levelname)s: {}{}%(message)s{}".format(
                                                                Colors.CYELLOW, Colors.ENDC, Colors.CBOLD, Colors.ENDC)
    dbg_fmt = prefix + "{}%(levelname)s (%(module)s: %(lineno)d): {}{}%(message)s{}".format(
                                                                Colors.CGREEN2, Colors.ENDC, Colors.CBOLD, Colors.ENDC)

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = ColorLogger.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = ColorLogger.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = ColorLogger.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = ColorLogger.wrn_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


class PlainLogger(logging.Formatter):
    """Class for plain white python logs for the application.

    Use if terminal does not support coloring.
    """

    prefix = '[%(name)s] - '
    err_fmt = prefix + "%(levelname)s (%(module)s: %(lineno)d): %(message)s"
    info_fmt = prefix + "%(levelname)s: %(message)s"
    wrn_fmt = prefix + "%(levelname)s: %(message)s"
    dbg_fmt = prefix + "%(levelname)s (%(module)s: %(lineno)d): %(message)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = PlainLogger.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = PlainLogger.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = PlainLogger.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = PlainLogger.wrn_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def file_handler(logfile):
    with open(ROOT_PATH + '/logs/log.log', 'w') as f:
        f.truncate(0)
    return logging.FileHandler(logfile)


def std_handler():
    return logging.StreamHandler(sys.stdout)


logger = logging.getLogger('Behavior-Log')

fmt = ColorLogger()
# stdout as output
hdlr = std_handler()
# file as output
# hdlr = file_handler(ROOT_PATH + '/logs/log.log')
hdlr.setFormatter(fmt)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
