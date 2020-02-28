#!/usr/bin/env python

# std libs
import os
import sys
import logging


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(filename='logfile', filemode='w', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(name)s: %(funcName)s: %(message)s')

formatter = logging.Formatter('%(asctime)s %(levelname)s: %(name)s: %(funcName)s: %(message)s')

handler_critical_event = logging.StreamHandler(sys.stdout)
handler_critical_event.setLevel(logging.CRITICAL)
handler_critical_event.setFormatter(formatter)

handler_error_event = logging.StreamHandler(sys.stdout)
handler_error_event.setLevel(logging.CRITICAL)
handler_error_event.setFormatter(formatter)


def getLogger(name):
    """Get a configured logger with specified name.

    Parameters
    ----------
    name : str
        Name of logger

    Returns
    -------
    logging.log instance
    """
    log = logging.getLogger(name)
    log.addHandler(handler_error_event)
    return(log)


def getClassName(self):
    return(self.__class__.__module__ + '.' + type(self).__name__)


# -----------------------------------------------------------------------------
# misc
# -----------------------------------------------------------------------------
def window_width():
    """Get width of terminal window

    Returns
    -------
    int
        Number of columns in window

    Raises
    ------
    OSError
        If unknown OSError.
    """
    try:
        return os.get_terminal_size().columns
    except OSError as e:
        if 'Errno 25' in str(e):
            return 80
        else:
            raise e


def string_truncate(s, mode='m'):
    """Truncate string to within terminal window width

    Parameters
    ----------
    s : str
        Line to print
    mode : str, optional
        Case spesific identifier

    Returns
    -------
    str
        Truncated string.
    """
    t = s
    width = window_width() - 11
    length = len(s)
    mid = int(length / 2)

    # special handling of '_reg' files
    if '_reg' in s and mode != 'r':
        width += 4
        mid = int((length - 4) / 2)

    if length <= width:
        return(s)
    diff = length - width

    if mode == 'r':
        t = s[:(-diff)] + ' ...'
    if mode == 'l':
        t = '... ' + s[diff:]
    if mode == 'm':
        d1 = int(diff / 2)
        d2 = diff - d1 + 1
        t = s[:(mid - d1)] + ' ... ' + s[mid + d2:]

    return(t)


def print_truncate(s, mode='m'):
    print(string_truncate(str(s), mode=mode))


def debug_start():
    print("################################################")


def debug_end():
    print("################################################")
    raise Exception("DEBUG end")
