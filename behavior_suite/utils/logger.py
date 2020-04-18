# Code from https://stackoverflow.com/a/8349076
import logging
import sys

from colors import Colors


class MyFormatter(logging.Formatter):

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
            self._fmt = MyFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = MyFormatter.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = MyFormatter.wrn_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result

class MyConsoleFormatter(logging.Formatter):

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
            self._fmt = MyConsoleFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = MyConsoleFormatter.info_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = MyConsoleFormatter.err_fmt

        elif record.levelno == logging.WARNING:
            self._fmt = MyConsoleFormatter.wrn_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


logger = logging.getLogger('Behavior-Log')
with open('/home/fran/github/BehaviorSuite/behavior_suite/logs/log.log', 'w') as f:
    f.truncate(0)

fmt = MyFormatter()
hdlr = logging.StreamHandler(sys.stdout)
# hdlr = logging.FileHandler('/home/fran/github/BehaviorSuite/behavior_suite/logs/log.log')
hdlr.setFormatter(fmt)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)
