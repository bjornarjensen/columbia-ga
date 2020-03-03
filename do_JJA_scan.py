#!/usr/bin/env python

# libs

# own modules
import modules.aos as aos
import modules.fmt as fmt
import modules.gen as gen
import modules.data as data
import modules.parallel as parallel
import modules.evaluate as evaluate


log = gen.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------------------------------
log.info('Initializing aos.')
aos.path = 'objects'
aos.verbose = 0
aos.auto = 1

# --------------------------------------------------------------------------------------------------
# prepare data and model names
# --------------------------------------------------------------------------------------------------
# List with names of models to exclude from calculations. Then filter.
log.info('Loading data.')
excluded_models = []
models = [m for m in data.NAMES_UA if m not in excluded_models]

# Load and format historic data
historic_month_1d = data.ua_ens_month_avg(time_period='1979-2005', models=models)
[historic_data_month_2d, historic_area_month_2d] = fmt.crop_2d_360_180_2d_360_60(
    fmt.reformat_1d_ens_2d_360_180(historic_month_1d))

# Load and format future data
future_month_1d = data.ua_ens_month_avg(time_period='2070-2099', models=models)
[future_data_month_2d, future_area_month_2d] = fmt.crop_2d_360_180_2d_360_60(
    fmt.reformat_1d_ens_2d_360_180(future_month_1d))

n_model = historic_month_1d.n_models
models = historic_month_1d._models

# --------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------
# Setting up the actual evaluation. `Evaluate` is the class computing the interesting results.
log.info('Preparing evaluation.')
e = evaluate.Evaluate(historic_data_month_2d, future_data_month_2d, n_model)
e = parallel.np_exec(e, axis=1, n_partitions=500)

# --------------------------------------------------------------------------------------------------
# ga
# --------------------------------------------------------------------------------------------------
# AOS settings - note aos.auto = 0 which means "recompute despite existing results".
aos.path = 'objects'
aos.verbose = 0
aos.auto = 0

# Launch the evaluation for all 360 degrees, with a minumum window
# of 10 degrees longitude and a maximum of 30 degrees longitude.
log.info('Launching evaluation.')
ea = evaluate.EvaluateAll(e, n=360, min_length=10, max_length=30)

# Retrieve and save results as a CSV formatted file.
log.info('Saving results.')
result_all = ea.result
result_all.to_csv('results_all.csv', index=False)

# Print a short sample with some of the results.
print('-' * 50)
print('Random sample from results:')
print(result_all.sample(10))
