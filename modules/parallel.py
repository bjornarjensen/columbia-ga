#!/usr/bin/env python

# std libs
import math
import codecs
import pickle
import importlib
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------------
def library_existing(lib):
    """Check if library exists.

    Parameters
    ----------
    lib : str
        name of library

    Returns
    -------
    bool
        True of library exists, else False
    """
    spec = importlib.util.find_spec(lib)
    return (spec is not None)


def data_frame_repartition(df, n, spark):
    df1 = df.repartition(n)
    df1.write.parquet("foo.parquet")
    df2 = spark.read.load("foo.parquet")
    return(df2)


# -----------------------------------------------------------------------------
# classes
# -----------------------------------------------------------------------------
class np_exec:
    def __init__(self, f, axis=0, n_partitions=-5):
        # admin
        self.f = f
        self.axis = axis
        self.t_exec = 0
        self.t_admin = 0
        self.t_df0 = 0
        self.t_df1 = 0
        self.t_df2 = 0
        self.t_df3 = 0
        self.n_eval = 0
        self.n_calls = 0
        self.n_partitions_sum = 0
        self.n_executors_sum = 0

        # check if spark library is existing
        if not library_existing("pyspark"):
            self.parallel = 0
            self.n_partitions = 1
        else:
            import os
            import pyspark
            from pyspark.sql import SparkSession

            self.parallel = 1
            self.f = f
            self.n_partitions = n_partitions

            self.spark = SparkSession.builder.appName('CreateSparkDF').getOrCreate()
            self.n_executors = int(self.spark.sparkContext._jsc.sc(
            ).getExecutorMemoryStatus().keySet().size()) - 1
            self.spark.sparkContext.setLogLevel("WARN")
            # os.environ['PROJ_LIB']='/opt/intel/intelpython3/share/proj'
            os.environ['MKL_NUM_THREADS'] = '1'

            if self.n_partitions < 0:
                self.n_partitions = -self.n_partitions * self.n_executors

    def __call__(self, x):
        import time

        # initialize timer
        t0 = time.time()

        # ensemble size (self.n_ens)
        self.n_ens = x.shape[self.axis]

        # admin
        self.n_calls += 1
        self.n_eval += self.n_ens
        self.n_partitions_sum += self.n_partitions

        # sequential execution
        if self.parallel == 0:
            y = self.f(x)
            self.t_exec += time.time() - t0
            return(y)

        # admin
        self.n_executors_sum += self.n_executors

        # parallel execution
        # load libraries
        from pyspark.sql.functions import udf

        # move ensemble column to dimension 0
        x0 = x
        if self.axis > 0:
            x0 = np.moveaxis(x, self.axis, 0)

        # partitions
        partitions_length_low = math.floor(self.n_ens / self.n_partitions)
        assert(partitions_length_low >= 1)
        partitions_length_high = partitions_length_low + 1
        n_partitions_high = self.n_ens - partitions_length_low * self.n_partitions
        n_partitions_low = self.n_partitions - n_partitions_high
        partitions_boundaries = [0] * (self.n_partitions + 1)
        for k in range(n_partitions_low):
            partitions_boundaries[k + 1] = partitions_boundaries[k] + partitions_length_low
        for k in range(n_partitions_low, self.n_partitions):
            partitions_boundaries[k + 1] = partitions_boundaries[k] + partitions_length_high
        x0_partitions = [None] * self.n_partitions
        for k in range(self.n_partitions):
            x0_partitions[k] = x0[partitions_boundaries[k]:partitions_boundaries[k + 1]]

        # time admin finished
        self.t_admin += time.time() - t0

        # udf
        f = self.f
        axis = self.axis

        @udf('string')
        def f_wrapper(x0_partition_pickled):

            # extract modules from transfer.tgz
            import subprocess
            from pyspark import SparkFiles
            t = SparkFiles.get('transfer.tgz')
            subprocess.call(['tar', 'xf', t])

            nonlocal f, axis

            x0_partition = pickle.loads(codecs.decode(x0_partition_pickled.encode(), "base64"))
            x_partition = np.moveaxis(x0_partition, 0, axis)
            y_partition = f(x_partition)
            y0_partition = np.moveaxis(y_partition, axis, 0)
            y0_partition_pickled = codecs.encode(pickle.dumps(y0_partition), "base64").decode()

            return(y0_partition_pickled)

        # create dataframe
        x0_partitions_pickled = [None] * self.n_partitions
        for k in range(self.n_partitions):
            x0_partitions_pickled[k] = codecs.encode(
                pickle.dumps(x0_partitions[k]), "base64").decode()
        columns = ['partition', 'x0_partitions_pickled']
        values = []
        for partition in range(self.n_partitions):
            values.append((partition, x0_partitions_pickled[partition]))
        df0 = self.spark.createDataFrame(values, columns)
        self.t_df0 += time.time() - t0
        df1 = df0.repartition(self.n_partitions)
        self.t_df1 += time.time() - t0
        df2 = df1.withColumn('y0_partitions_pickled', f_wrapper(df1.x0_partitions_pickled))
        self.t_df2 += time.time() - t0
        df3 = df2.toPandas()
        self.t_df3 += time.time() - t0
        self.t_exec = self.t_df3

        # assemble result
        y0_partitions = [None] * self.n_partitions
        for index, row in df3.iterrows():
            y0_partitions[row['partition']] = pickle.loads(
                codecs.decode(row['y0_partitions_pickled'].encode(), "base64"))
        y0 = np.vstack(y0_partitions)
        if self.axis == 0:
            y = y0
        else:
            y = np.moveaxis(y0, 0, self.axis)

        return(y)

    def __str__(self):
        s = '## np_exec ##\n'
        s += "n_calls = %d\n" % self.n_calls
        s += "n_eval = %d\n" % self.n_eval
        s += "axis = %d\n" % self.axis
        s += "exec time: %f\n" % self.t_exec
        s += "exec time per eval: %f\n" % (self.t_exec / self.n_eval)

        s += "parallel = %d\n" % self.parallel
        if self.parallel:
            s += "n_executors (avg) = %f\n" % (self.n_executors_sum / self.n_calls)
            s += "n_partitions (avg) = %f\n" % (self.n_partitions_sum / self.n_calls)
            s += "times:\n"
            s += "   admin: %f\n" % self.t_admin
            s += "   df0:   %f\n" % self.t_df0
            s += "   df1:   %f\n" % self.t_df1
            s += "   df2:   %f\n" % self.t_df2
            s += "   df3:   %f\n" % self.t_df3
        return(s)
