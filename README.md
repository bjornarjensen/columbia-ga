# COLUMBIA

COLUMBIA is an interdisciplinary project that aims to develop an innovative tool, based on the state-of-the-art machine learning technology, to efficiently analyze large amount of model data to better understand
why some models behave very differently than the others. Combined with our current knowledge of how the climate system works based on the observational evidence, we will constraint the large spread in these model simulations. Also using our novel tool, we hope to filter out those climate models that do not represent well the important dynamics observed in nature.

It is funded by Research Council of Norway (RCN) and led by **Research Professor Jerry Tjiputra** at NORCE - Norwegian Research Centre AS.

# Authors

The main developer is **Chief Scientist Klaus Johannsen**, also at NORCE. 

Contributing developers:

- Bjørnar Jensen, Senior researcher at NORCE

# Usage

On Spark cluster:
```./run do_JJA_scan.py <ncpus>```

On computer without spark:
```python3 do_JJA_scan.py```


# Modification

- Edit/create configuration files that matches your data storage location.
- Update/create data class in `modules/data.py` (see `data.ua_ens_month_avg` for an example).

## Simple cases - no genetic algorithm

Here we have two classed, defined in the `do_JJA_scan.py`. `Evaluate` which carries out the evaluation through its `__call__` member function. `Evaluate_all` which creates numpy arrays that mimic genetic algorithm populations. This then calls `evaluate` on the population, gathers the result and saves the computed output to files.

## With genetic algorithms

Create a class `Evaluate` with a `__call__` member function. Pass this as `eval_` parameter into the genetic algorithm, see `modules.ga.Simple` for an example. Adjust genetic algorithm to suit your needs.

