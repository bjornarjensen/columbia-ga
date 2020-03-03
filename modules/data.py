#!/usr/bin/env python

# std libs
import cftime
import os.path
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import pathlib as pl

import modules.aos as aos
import modules.gen as gen

# -----------------------------------------------------------------------------
# model names
# -----------------------------------------------------------------------------
# uas
NAMES_UAS = [
    'CMCC-CESM', 'CMCC-CM', 'CMCC-CMS', 'CNRM-CM5', 'GFDL-CM3', 'GFDL-ESM2G', 'GFDL-ESM2M',
    'HadGEM2-AO', 'HadGEM2-CC', 'HadGEM2-ES', 'inmcm4', 'IPSL-CM5A-LR', 'IPSL-CM5A-MR',
    'IPSL-CM5B-LR', 'MIROC5', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'MPI-ESM-LR', 'MPI-ESM-MR',
    'MRI-CGCM3', 'MRI-ESM1', 'NorESM1-M']

# ua
NAMES_UA = [
    'ACCESS1-0', 'BNU-ESM', 'CESM1-CAM5', 'CNRM-CM5', 'GFDL-CM3', 'GISS-E2-H-CC', 'HadGEM2-CC',
    'IPSL-CM5A-MR', 'MIROC-ESM-CHEM', 'MRI-ESM1', 'ACCESS1-3', 'CanESM2', 'CMCC-CESM',
    'CSIRO-Mk3-6-0', 'GFDL-ESM2G', 'GISS-E2-R', 'HadGEM2-ES', 'IPSL-CM5B-LR', 'MPI-ESM-LR',
    'NorESM1-M', 'bcc-csm1-1', 'CCSM4', 'CMCC-CM', 'FGOALS-g2', 'GFDL-ESM2M', 'GISS-E2-R-CC',
    'inmcm4', 'MIROC5', 'MPI-ESM-MR', 'NorESM1-ME', 'bcc-csm1-1-m', 'CESM1-BGC', 'CMCC-CMS',
    'FIO-ESM', 'GISS-E2-H', 'HadGEM2-AO', 'IPSL-CM5A-LR', 'MIROC-ESM', 'MRI-CGCM3']

# -----------------------------------------------------------------------------
# global variables
# -----------------------------------------------------------------------------
DATA_UNINITIALIZED = 3.1415e14


# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------
def numpy_array_hash(a):
    b = a.reshape(-1)
    b = b[~np.isnan(b)]
    w = np.linspace(1, 2, b.shape[0], endpoint=True).reshape(-1)
    return(hash(np.dot(b, w)))


def time_datetime64_to_month(t):
    # defining Jan 1971 as month 0
    if isinstance(t[0], np.datetime64):
        tt = pd.to_datetime(t)
        month = (tt.year - 1971) * 12 + (tt.month - 1)
        return(np.array(month, dtype=int))

    if isinstance(t[0], cftime._cftime.DatetimeNoLeap):
        tt = np.zeros((t.shape[0]), dtype=int)
        for i in range(t.shape[0]):
            tt[i] = (t[i].year - 1971) * 12 + (t[i].month - 1)
        return(tt)

    if isinstance(t[0], datetime.datetime):
        tt = np.zeros((t.shape[0]), dtype=int)
        for i in range(t.shape[0]):
            tt[i] = (t[i].year - 1971) * 12 + (t[i].month - 1)
        return(tt)

    if isinstance(t[0], cftime._cftime.Datetime360Day):
        tt = np.zeros((t.shape[0]), dtype=int)
        for i in range(t.shape[0]):
            tt[i] = (t[i].year - 1971) * 12 + (t[i].month - 1)
        return(tt)

    if isinstance(t[0], cftime._cftime.DatetimeGregorian):
        tt = np.zeros((t.shape[0]), dtype=int)
        for i in range(t.shape[0]):
            tt[i] = (t[i].year - 1971) * 12 + (t[i].month - 1)
        return(tt)

    if isinstance(t[0], cftime._cftime.DatetimeProlepticGregorian):
        tt = np.zeros((t.shape[0]), dtype=int)
        for i in range(t.shape[0]):
            tt[i] = (t[i].year - 1971) * 12 + (t[i].month - 1)
        return(tt)

    raise Exception("unknown type '%s' for time detected" % type(t[0]))


def array_create_unitialized(t):
    return(np.full(t, DATA_UNINITIALIZED))


def array_is_completely_uninitialized(x):
    return(np.min(x) == DATA_UNINITIALIZED)


def array_is_initialized(x):
    return(np.nanmax(x) < DATA_UNINITIALIZED)


# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
class time_avg_lat_lon:
    def __init__(self, data, time_period=0):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("starting")

        # input
        if time_period == '1971-2000':
            self._time_period = '1971-2000'
            self._n_month = 360
            self._month_offset = 0
        elif time_period == '2071-2100':
            self._time_period = '2071-2100'
            self._n_month = 360
            self._month_offset = 100 * 12
        elif time_period == '1979-2005':
            self._time_period = '1979-2005'
            self._n_month = 27 * 12
            self._month_offset = 8 * 12
        elif time_period == '2070-2099':
            self._time_period = '2070-2099'
            self._n_month = 360
            self._month_offset = 99 * 12
        else:
            log.critical("Initialization error, provide time_period")
            raise Exception("time_avg_lat_lon: Initialization error, provide time_period")
        if not data:
            log.critical("Initialization error, provide data")
            raise Exception("time_avg_lat_lon: Initialization error, provide data")

        # initialize
        log.info("initializing from %s" % data[0])
        values = None
        if (data[0] == 'uas list') or (data[0] == 'ua list'):
            # initialize from list
            for list_item in data[1]:
                if values is None:
                    values = array_create_unitialized((self._n_month, list_item.ny, list_item.nx))
                    self._x = list_item.x
                    self._y = list_item.y
                if not data[1][0].grid_equal(list_item):
                    log.critical("initialization error, grid mismatch")
                    raise Exception("time_avg_lat_lon: Initialization error, grid mismatch")
                for t_idx in range(list_item.nt):
                    month = list_item.t[t_idx] - self._month_offset
                    if (month < 0) or (month >= self._n_month):
                        continue
                    if not array_is_completely_uninitialized(values[month]):
                        log.error("array not completely uninitialized")
                        print("WARNING: time_avg_lat_lon: array not completely uninitialized")
                    values[month] = list_item.data[t_idx]
        else:
            log.critical("initialization mode '%s' not recognized" % data[0])
            raise Exception("time_avg_lat_lon: initialization mode '%s' not recognized" % data[0])

        if not array_is_initialized(values):
            log.critical(
                "Initialization error, array not completely initialized from list %s" % str(data[1]))
            raise Exception(
                "time_avg_lat_lon: Initialization error, array not completely initialized")
        values = np.mean(values, axis=0)

        # output array
        nx = self._x.shape[0]
        ny = self._y.shape[0]
        self.points = np.zeros((nx * ny, 2))
        self.data = np.zeros((nx * ny, 1))
        self.area = np.zeros((nx * ny, 1))
        n = 0
        for i in range(nx):
            for j in range(ny):
                self.points[n, 0] = self._x[i]
                self.points[n, 1] = self._y[j]
                self.data[n, 0] = values[j, i]
                self.area[n, 0] = np.cos(np.pi / 180 * self._y[j]) * np.sin(np.pi / 360) / 360
                n += 1

        log.info("finished loading data from list")

    def _grid_hash(self):
        h = hash("time_avg_lat_lon")
        h += numpy_array_hash(self.points)
        return(h)

    def __hash__(self):
        h = hash("time_avg_lat_lon")
        h += numpy_array_hash(self.points)
        h += numpy_array_hash(self.data)
        return(h)

    def __eq__(self, other):
        if self.__hash__() == other.__hash__():
            return(True)
        return(False)

    def grid_equal(self, other):
        if self._grid_hash() == other._grid_hash():
            return(True)
        return(False)


class time_avg_lat_lon_dir(time_avg_lat_lon):
    def __init__(self, directory, time_period=0):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("starting")

        # aos
        obj = aos.DataObject(('time_avg_lat_lon_dir', directory, time_period))
        if obj.load():
            obj.copy(self)
            log.info("loaded")
            return(None)

        ua_list = []
        for f in directory.iterdir():
            if '_reg.nc' in str(f):
                gen.print_truncate("  Loading file %s" % f)
                ua_list = ua_list + [ua(['file', directory.joinpath(f)])]
        super(time_avg_lat_lon_dir, self).__init__(['ua list', ua_list], time_period=time_period)

        # aos
        obj.save(self)
        log.info("created")


class time_avg_month_lat_lon:
    def __init__(self, data, time_period=0):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("starting")

        # input
        if time_period == '1971-2000':
            self._time_period = '1971-2000'
            self._n_month = 360
            self._month_offset = 0
        elif time_period == '2071-2100':
            self._time_period = '2071-2100'
            self._n_month = 360
            self._month_offset = 100 * 12
        elif time_period == '1979-2005':
            self._time_period = '1979-2005'
            self._n_month = 27 * 12
            self._month_offset = 8 * 12
        elif time_period == '2070-2099':
            self._time_period = '2070-2099'
            self._n_month = 360
            self._month_offset = 99 * 12
        else:
            log.critical("Initialization error, provide time_period")
            raise Exception("time_avg_month_lat_lon: Initialization error, provide time_period")
        if not data:
            log.critical("Initialization error, provide data")
            raise Exception("time_avg_month_lat_lon: Initialization error, provide data")

        # initialize
        log.info("initializing from %s" % data[0])
        values = None
        if data[0] == 'ua list':
            # initialize from list
            for list_item in data[1]:
                if values is None:
                    values = array_create_unitialized((self._n_month, list_item.ny, list_item.nx))
                    self._x = list_item.x
                    self._y = list_item.y
                if not data[1][0].grid_equal(list_item):
                    log.critical("initialization error, grid mismatch")
                    raise Exception("time_avg_month_lat_lon: Initialization error, grid mismatch")
                for t_idx in range(list_item.nt):
                    month = list_item.t[t_idx] - self._month_offset
                    if (month < 0) or (month >= self._n_month):
                        continue
                    if not array_is_completely_uninitialized(values[month]):
                        log.error("array not completely uninitialized")
                        print("WARNING: time_avg_month_lat_lon: array not completely uninitialized")
                    values[month] = list_item.data[t_idx]
        else:
            log.critical("initialization mode '%s' not recognized" % data[0])
            raise Exception(
                "time_avg_month_lat_lon: initialization mode '%s' not recognized" % data[0])

        if not array_is_initialized(values):
            log.critical(
                "Initialization error, array not completely initialized from list %s" % str(data[1]))
            raise Exception(
                "time_avg_lat_lon: Initialization error, array not completely initialized")

        # create month-averages for 24 month
        # first 12 month: year 0 ... n-1
        # second 12 month: year 1 ... n
        n_month = values.shape[0]
        n_years = int(n_month / 12)
        assert(12 * n_years == n_month)
        values = values.reshape(n_years, 12, 180, 360)
        values_1 = np.mean(values[:-1], axis=0)
        values_2 = np.mean(values[1:], axis=0)
        values = np.vstack((values_1, values_2))

        # output array
        nx = self._x.shape[0]
        ny = self._y.shape[0]
        self.points = np.zeros((nx * ny, 2))
        self.data = np.zeros((nx * ny, 24))
        self.area = np.zeros((nx * ny, 1))
        n = 0
        for i in range(nx):
            for j in range(ny):
                self.points[n, 0] = self._x[i]
                self.points[n, 1] = self._y[j]
                self.data[n, :] = values[:, j, i]
                self.area[n, 0] = np.cos(np.pi / 180 * self._y[j]) * np.sin(np.pi / 360) / 360
                n += 1

        log.info("finished loading data from list")

    def _grid_hash(self):
        h = hash("time_avg_lat_lon")
        h += numpy_array_hash(self.points)
        return(h)

    def __hash__(self):
        h = hash("time_avg_lat_lon")
        h += numpy_array_hash(self.points)
        h += numpy_array_hash(self.data)
        return(h)

    def __eq__(self, other):
        if self.__hash__() == other.__hash__():
            return(True)
        return(False)

    def grid_equal(self, other):
        if self._grid_hash() == other._grid_hash():
            return(True)
        return(False)


class time_avg_month_lat_lon_dir(time_avg_month_lat_lon):
    def __init__(self, directory, time_period=0):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("starting")

        # aos
        obj = aos.DataObject(('time_avg_month_lat_lon_dir', directory, time_period))
        if obj.load():
            obj.copy(self)
            log.info("loaded")
            return(None)

        ua_list = []
        for f in directory.iterdir():
            if '_reg.nc' in str(f):
                gen.print_truncate("  Loading file %s" % f)
                ua_list = ua_list + [ua(['file', directory.joinpath(f)])]
        super(time_avg_month_lat_lon_dir, self).__init__(
            ['ua list', ua_list], time_period=time_period)

        # aos
        obj.save(self)
        log.info("created")


# -----------------------------------------------------------------------------
# uas
# -----------------------------------------------------------------------------
class uas:
    def __init__(self, data):
        if not data:
            raise Exception("uas: Initialization error, provide data")
        if data[0] == 'file':
            self._init_uas_from_file(data[1])
        else:
            raise Exception("uas: Initialization error, unknow mode")

    def _init_uas_from_file(self, uas_file):
        if not os.path.isfile(uas_file):
            raise Exception("uas: Initialization error, file '%s' not found" % uas_file)
        self._uas_file = uas_file
        g = xr.open_dataset(self._uas_file)

        # store grid
        self.x = g['lon'].values.reshape(-1)
        self.y = g['lat'].values.reshape(-1)
        self.z = g['height'].values.reshape(-1)
        self.t = time_datetime64_to_month(g['time'].values)
        self.nx = self.x.shape[0]
        self.ny = self.y.shape[0]
        self.nz = self.z.shape[0]
        if self.nz != 1:
            raise Exception("uas: Initialization error, only one z-level allowed")
        self.nt = self.t.shape[0]

        # store data
        self.data = g['uas'].values

    def attributes(self):
        return(['_uas_file', 'x', 'y', 'z', 't', 'nx', 'ny', 'nz', 'nt'])

    def __str__(self):
        s = "--- uas ---\n"
        s = s + "file: %s\n" % self._uas_file
        s = s + "nx: %d\n" % self.nx
        s = s + "ny: %d\n" % self.ny
        s = s + "nz: %d\n" % self.nz
        s = s + "nt: %d\n" % self.nt
        return(s)

    def _grid_hash(self):
        h = hash("uas")
        h += numpy_array_hash(self.x)
        h += numpy_array_hash(self.y)
        h += numpy_array_hash(self.z)
        return(h)

    def __hash__(self):
        h = hash("uas")
        h += numpy_array_hash(self.x)
        h += numpy_array_hash(self.y)
        h += numpy_array_hash(self.z)
        h += numpy_array_hash(self.t)
        h += numpy_array_hash(self.data)
        return(h)

    def grid_equal(self, other):
        if self._grid_hash() == other._grid_hash():
            return(True)
        return(False)

    def __eq__(self, other):
        if self.__hash__() == other.__hash__():
            return(True)
        return(False)


class uas_ens_avg:
    def __init__(self, time_period=0):
        # aos
        obj = aos.DataObject(('uas_ens_avg', time_period))
        if obj.load():
            obj.copy(self)
            return(None)

        # init
        self._time_period = time_period
        self._models = NAMES_UAS

        # create list of time_avg_lat_lon 2D-fields
        ensemble_time_avg_lat_lon = []
        if (time_period == '1971-2000') or (time_period == '1979-2005'):
            directory_prefix = 'data/ModData2/CMIP5/atmos/historical/uas/mon/'
        elif (time_period == '2071-2100') or (time_period == '2070-2099'):
            directory_prefix = 'data/ModData2/CMIP5/atmos/rcp85/uas/mon/'
        else:
            raise Exception("uas_ens_avg: time period '%s' not known" % time_period)
        print("Load data from time period %s" % time_period)

        for name in self._models:
            print("Loading %s from file" % name)
            directory = directory_prefix + name + '/r1i1p1/'
            uas_list = [uas(['file', os.path.join(directory, f)])
                        for f in os.listdir(directory) if f.endswith(".nc")]
            tall = time_avg_lat_lon(['uas list', uas_list], time_period=time_period)
            ensemble_time_avg_lat_lon += [tall]

        # check consistancy of grids in the ensemble of time_avg_lat_lon 2D-fields and store grid
        tall_0 = ensemble_time_avg_lat_lon[0]
        for tall in ensemble_time_avg_lat_lon:
            if not tall_0.grid_equal(tall):
                raise Exception("uas_ens_avg: error, grids are not consistent")
        self.points = tall_0.points
        self.area = tall_0.area

        # put ensemble into a 2D array: [data,talls]
        data = []
        for tall in ensemble_time_avg_lat_lon:
            data += [tall.data]
        self.data = np.hstack(data)

        # area

        # aos
        obj.save(self)

    def __str__(self):
        s = "--- uas_ens_avg ---\n"
        s = s + "Models: [%s]\n" % (', '.join(self._models))
        s = s + "time period: %s\n" % self._time_period
        s = s + "points: %d\n" % self.points.shape[0]
        s = s + "data: (%d, %d)\n" % (self.data.shape)
        return(s)


# -----------------------------------------------------------------------------
# ua
# -----------------------------------------------------------------------------
class ua:
    def __init__(self, data):
        # aos
        obj = aos.DataObject(('ua', data))
        if obj.load():
            obj.copy(self)
            return(None)

        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("creating class")

        if not data:
            raise Exception("ua: Initialization error, provide data")
        if data[0] == 'file':
            self._init_ua_from_file(data[1])
        else:
            raise Exception("ua: Initialization error, unknow mode")

        # aos
        obj.save(self)

    def _init_ua_from_file(self, ua_file):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("loading data from file '%s'" % ua_file)

        if not ua_file.exists():
            log.critical("Initialization error, file '%s' not found" % ua_file)
            raise Exception("ua: Initialization error, file '%s' not found" % ua_file)

        self._ua_file = ua_file
        g = xr.open_dataset(self._ua_file)
        self.plev_to_extract = 70000

        # store grid
        self.x = g['lon'].values.reshape(-1)
        self.y = g['lat'].values.reshape(-1)
        if 'plev' in g:
            self.plev = g['plev'].values.reshape(-1)
        elif 'lev' in g:
            self.plev = g['lev'].values.reshape(-1)
        else:
            log.critical("Initialization error, cannot read 'plev'")
            raise Exception("ua: Initialization error, cannot read 'plev'")

        log.info("plev: %s" % str(self.plev))
        idc = np.where(self.plev == self.plev_to_extract)[0]
        if idc.shape[0] != 1:
            log.critical("ua: Initialization error: cannot find plevel %f" % self.plev_to_extract)
            raise Exception("ua: Initialization error: cannot find plevel %f" %
                            self.plev_to_extract)
        self.plev_idx = idc[0]
        self.t = time_datetime64_to_month(g['time'].values)
        self.nx = self.x.shape[0]
        self.ny = self.y.shape[0]
        self.nt = self.t.shape[0]

        # store data
        self.data = g['ua'].values[:, self.plev_idx, :, :]  # array for (time, lat, lon)

    def attributes(self):
        return(['_ua_file', 'plev_to_extract', 'x', 'y',
                'plev', 'plev_idx', 't', 'nx', 'ny', 'nt'])

    def __str__(self):
        s = "--- ua ---\n"
        s = s + "file: %s\n" % self._uas_file
        s = s + "nx: %d\n" % self.nx
        s = s + "ny: %d\n" % self.ny
        s = s + "nt: %d\n" % self.nt
        return(s)

    def _grid_hash(self):
        h = hash("uas")
        h += numpy_array_hash(self.x)
        h += numpy_array_hash(self.y)
        h += numpy_array_hash(self.plev)
        return(h)

    def __hash__(self):
        h = hash("uas")
        h += numpy_array_hash(self.x)
        h += numpy_array_hash(self.y)
        h += numpy_array_hash(self.plev)
        h += numpy_array_hash(self.t)
        h += numpy_array_hash(self.data)
        h += hash(self.plev_to_extract)
        return(h)

    def grid_equal(self, other):
        if self._grid_hash() == other._grid_hash():
            return(True)
        return(False)

    def __eq__(self, other):
        if self.__hash__() == other.__hash__():
            return(True)
        return(False)


class ua_ens_avg:
    def __init__(self, cfg, time_period=0):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("start creating ua ensemble average")

        # aos
        obj = aos.DataObject(('ua_ens_avg', time_period))
        if obj.load():
            obj.copy(self)
            return(None)

        # init
        self._time_period = time_period
        self._models = NAMES_UA
        self.cfg = dict(cfg)

        # create list of time_avg_lat_lon 2D-fields
        ensemble_time_avg_lat_lon = []
        if time_period == '1979-2005':
            directory_prefix = pl.Path(
                self.cfg['data_folders']['historical']).joinpath('ua/mon')
        elif time_period == '2070-2099':
            directory_prefix = pl.Path(
                self.cfg['data_folders']['rcp85']).joinpath('ua/mon')
        else:
            raise Exception("ua_ens_avg: time period '%s' not known" % time_period)

        print("Load data from time period %s" % time_period)
        log.info("ua_ens_avg: Load data from time period %s" % time_period)
        for name in self._models:
            print("Loading %s from file" % name)
            log.info("Loading %s from file" % name)
            directory = directory_prefix.joinpath(name).joinpath('r1i1p1')
            tall = time_avg_lat_lon_dir(directory, time_period=time_period)
            ensemble_time_avg_lat_lon += [tall]

        # check consistancy of grids in the ensemble of time_avg_lat_lon 2D-fields and store grid
        tall_0 = ensemble_time_avg_lat_lon[0]
        for tall in ensemble_time_avg_lat_lon:
            if not tall_0.grid_equal(tall):
                raise Exception("uas_ens_avg: error, grids are not consistent")
        self.points = tall_0.points
        self.area = tall_0.area

        # put ensemble into a 2D array: [data,talls]
        data = []
        for tall in ensemble_time_avg_lat_lon:
            data += [tall.data]
        self.data = np.hstack(data)

        # area

        # aos
        obj.save(self)

    def __str__(self):
        s = "--- uas_ens_avg ---\n"
        s = s + "Models: [%s]\n" % (', '.join(self._models))
        s = s + "time period: %s\n" % self._time_period
        s = s + "points: %d\n" % self.points.shape[0]
        s = s + "data: (%d, %d)\n" % (self.data.shape)
        return(s)


class ua_ens_month_avg:
    def __init__(self, cfg, time_period=0, models=NAMES_UA):
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("start creating ua ensemble monthly average")

        # aos
        obj = aos.DataObject(('ua_ens_month_avg', time_period, models))
        if obj.load():
            obj.copy(self)
            return(None)

        # init
        self._time_period = time_period
        self._models = models
        self.cfg = dict(cfg)

        # create list of time_avg_lat_lon 2D-fields
        ensemble_time_avg_lat_lon = []
        if time_period == '1979-2005':
            directory_prefix = pl.Path(
                self.cfg['data_folders']['historical']).joinpath('ua/mon')
        elif time_period == '2070-2099':
            directory_prefix = pl.Path(
                self.cfg['data_folders']['rcp85']).joinpath('ua/mon')
        else:
            raise Exception("ua_ens_avg: time period '%s' not known" % time_period)

        print("Load data from time period %s" % time_period)
        log.info("ua_ens_month_avg: Load data from time period %s" % time_period)
        for name in self._models:
            print("Loading %s from file" % name)
            log.info("Loading %s from file" % name)
            directory = directory_prefix.joinpath(name).joinpath('r1i1p1')
            tall = time_avg_month_lat_lon_dir(directory, time_period=time_period)
            ensemble_time_avg_lat_lon += [tall]

        # check consistancy of grids in the ensemble of time_avg_lat_lon 2D-fields and store grid
        tall_0 = ensemble_time_avg_lat_lon[0]
        for tall in ensemble_time_avg_lat_lon:
            if not tall_0.grid_equal(tall):
                raise Exception("ua_ens_month_avg: error, grids are not consistent")
        self.points = tall_0.points
        self.area = tall_0.area

        # put ensemble into a 3D array with shape (n_points, 24, n_models)
        tall_0 = ensemble_time_avg_lat_lon[0]
        n_points = tall_0.data.shape[0]
        self.n_models = len(ensemble_time_avg_lat_lon)
        self.data = np.zeros((n_points, 24, self.n_models))
        for tall, model in zip(ensemble_time_avg_lat_lon, range(self.n_models)):
            self.data[:, :, model] = tall.data

        # aos
        obj.save(self)

    def __str__(self):
        s = "--- uas_ens_month_avg ---\n"
        s = s + "Models: [%s]\n" % (', '.join(self._models))
        s = s + "time period: %s\n" % self._time_period
        s = s + "points: %d\n" % self.points.shape[0]
        s = s + "data: (%d, %d)\n" % (self.data.shape)
        return(s)
