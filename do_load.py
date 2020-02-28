#!/usr/bin/env python
"""Simple run to load and process aos objects without further tasks.

Creation of aos.DataObjects from raw data files can be quite time
intensive. This task can also be performed prior to actual processing
runs. This is an example of such a session.
"""

# own modules
import modules.aos as aos
import modules.fmt as fmt
import modules.data as data
import modules.evaluate as evaluate


# -----------------------------------------------------------------------------
# init
# -----------------------------------------------------------------------------
aos.path = 'objects'
aos.verbose = 0
aos.auto = 1

# -----------------------------------------------------------------------------
# prepare
# -----------------------------------------------------------------------------
historic_1d = data.ua_ens_avg(time_period='1979-2005')
[historic_data_2d, historic_area_2d] = fmt.crop_2d_360_180_2d_360_60(
    fmt.reformat_1d_ens_2d_360_180(historic_1d))
historic_data_2d_lon_avg = fmt.avg_lon(historic_data_2d)
historic_lm = evaluate.max_lat(historic_data_2d_lon_avg)

future_1d = data.ua_ens_avg(time_period='2070-2099')
[future_data_2d, future_area_2d] = fmt.crop_2d_360_180_2d_360_60(
    fmt.reformat_1d_ens_2d_360_180(future_1d))
future_data_2d_lon_avg = fmt.avg_lon(future_data_2d)
future_lm = evaluate.max_lat(future_data_2d_lon_avg)
