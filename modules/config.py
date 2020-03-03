#!/usr/bin/env python
"""Module to read and process yaml-formatted config files."""

# std libs
import yaml
import pathlib
import modules.gen as gen


# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------
def read(cfile):
    """Read `yaml` formatted config file.

    Parameters
    ----------
    cfile : str
        filename of config to read

    Returns
    -------
    dict
        Dictionary with config options.
    """
    log = gen.getLogger(__name__)

    # read config file
    cfile = pathlib.Path(cfile).resolve()
    if not cfile.exists():
        raise IOError('Config file does not exist: {}'.format(str(cfile)))

    with open(cfile, 'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)

    # check sections
    # Note that this only really give settings of where to find files.
    # Thus the key `data_folders` can be substituted with any other system
    # name, as long as it corresponds to the section in the config file.
    if 'data_folders' not in conf.keys():
        log.critical("'data_folders' not specified in config file '%s'" % cfile)

    return(conf)
