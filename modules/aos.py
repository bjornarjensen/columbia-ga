#!/usr/bin/env python

"""Module to handle data object storage.

This allows for faster processing on later runs as the data have been
extracted and saved in a more suitable format for further processing.

NOTE: Consistent use of this requires a fixed PYTHONHASHSEED environmental
variable set.

Attributes
----------
auto : int
    If 1 then automatically load existing data objects if found or create
    new ones after loading data if data object does not exist.
footline : str
    Separation line for visually identifying end of output from aos.DataObject.
headline : str
    Separation line for identifying start of output from aos.DataObject.
path : str
    Default path to data object storage
readonly : int
    If `readonly` is 1 then existing data objects on disk will not be modified.
    If 0 data objects will be updated with output from calculations.
verbose : int
    Add extra output. Valid range [0, 2]. Higher number, more output.
"""

import os
import pickle
import os.path
import numpy as np
import modules.gen as gen

# -----------------------------------------------------------------------------
# global variables
# -----------------------------------------------------------------------------
auto = 1
readonly = 0
path = 'objects'
verbose = 1
headline = '################# aos ############################'
footline = ''

# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------


def hsh(x):
    """Calculate hash of c

    Parameters
    ----------
    x :
        Input to hash

    Returns
    -------
    int
        Hash of x
    """
    if x is None:
        return(hash('None'))
    h = 0
    for i in range(len(x)):
        if x[i] is None:
            h += hash('None')
        elif isinstance(x[i], np.ndarray):
            a = x[i].reshape(-1)
            b = np.linspace(1, 2, a.shape[0])
            h += hash(np.sum(a * a * b))
        elif isinstance(x[i], list):
            h = 0
            for k in range(len(x)):
                h = h + k * hsh(x[k])
        else:
            h += hash(x[i])
    return(h)


def format_list(x):
    """Format a list into a string

    Parameters
    ----------
    x : TYPE
        List of items to format into string

    Returns
    -------
    str
        The input represented as a string.
    """
    L = list(map(lambda x: isinstance(x, str), x))
    s = "("
    for i in range(len(x)):
        if L[i]:
            s += "'" + x[i] + "'"
        else:
            s += '...'
        if i < len(x):
            s += ','
    s += ')'
    return(s)


def init(auto=1, readonly=0, path='objects', verbose=1):
    """Initialize global variables

    Parameters
    ----------
    auto : int, optional
        If 1 create new objects if needed, load existing ones.
        Else only create new files, do not load existing ones.
    readonly : int, optional
        Update existing object files with new results?
    path : str, optional
        Path to folder containing data objects.
    verbose : int, optional
        Select output level, 0-2, higher number - more output.
    """
    g_ = globals()
    g_['auto'] = auto
    g_['readonly'] = readonly
    g_['verbose'] = verbose
    g_['path'] = path



# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
class DataObject:

    """DataObject is a class for storing data as pickles.

    Based on the data used to create the data objects it calculates
    hash values which are used to identify the dataset on reload.
    This enables faster processing as extracted and filtered data
    can be reused. Results from previous runs will also be stored
    in the object files if `readonly` is not set. Thus retrieval
    of past results/runs is achievable.

    Attributes
    ----------
    X :
        Data, files, list of files, etc to load/retrieve.
    """

    def __init__(self, x):
        self.log = gen.getLogger(gen.getClassName(self))
        self.X = None
        self._x_ = x
        self._hash_ = hsh(x)
        self._file_ = "%s/data-%s" % (path, self._hash_)
        self._file_exists_ = os.path.isfile(self._file_)

        if verbose == 2:
            print(headline)
            print("auto=%d" % auto)
            print("action: init")
            print(format_list(self._x_))
            print("file exists: %d" % self._file_exists_)
            print("file=%s" % self._file_)
            print(footline)

        if 'PYTHONHASHSEED' not in os.environ.keys():
            self.log.warning('`PYTHONHASHSEED` environmental variable not set.' +
                             ' This will prevent `aos.DataObject` from identifying' +
                             ' existing files as each run will have new seeds.')

    def load(self):
        """Read from data object storage if file exists and auto != 0

        Returns
        -------
        bool
            True if file exists and auto != 0, else False
        """

        if auto and self._file_exists_:
            # Read from existing file
            f = open(self._file_, 'rb')
            self.X = pickle.load(f)

            self.log.info("read from file '%s'" % self._file_)
            if verbose == 1:
                print("Loading %s" % format_list(self._x_))
        else:
            # Will create new file
            self.log.info("creating file '%s'" % self._file_)
            if verbose == 1:
                print("Creating %s" % format_list(self._x_))

        if verbose == 2:
            print(headline)
            print("auto=%d" % auto)
            print("action: load")
            print("file exists: %d" % self._file_exists_)

            if auto and self._file_exists_:
                print("loading file '%s'" % self._file_)
            print(footline)

        return(auto and self._file_exists_)

    def copy(self, x):
        """Copy content read from pickle to x.

        Parameters
        ----------
        x : aos.DataObject
            DataObject to copy data to

        Raises
        ------
        e
            AttributeError if no data in DataObject.
        """
        try:
            x.__dict__ = self.X.__dict__.copy()
        except AttributeError as e:
            self.log.exception('No data object to copy: {}'.format(e))
            raise e
        if verbose == 2:
            print(headline)
            print("auto=%d" % auto)
            print("action: copy")
            print(footline)

    def save(self, x):
        """Save data to pickle

        Parameters
        ----------
        x :
            Data to write to pickle
        """
        global readonly
        if auto:
            f = open(self._file_, 'wb')
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
            if readonly:
                os.chmod(self._file_, 0o444)
        if verbose == 2:
            print(headline)
            print("auto=%d" % auto)
            print("action: save")
            print("saving file '%s'" % self._file_)
            print(footline)
