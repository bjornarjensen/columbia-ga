#!/usr/bin/env python
"""Module to read and process yaml-formatted config files."""

# std libs
import yaml
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
    cfile = 'conf/' + cfile
    with open(cfile, 'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)

    # check sections
    # Note that this only really give settings of where to find files.
    # Thus the key `hadoop` can be substituted with any other system name,
    # as long as it corresponds to the section in the config file.
    if 'hadoop' not in conf.keys():
        log.critical("'hadoop' not specified in config file '%s'" % cfile)

    return(conf)
