#!/usr/bin/env python

# std libs
# import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
# from deap import algorithms
from deap.algorithms import *
# import deap.cma as cma


# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------
def _pop_to_nparray(p):
    return(np.transpose(np.array(p)))


def _popfitness_to_nparray(p):
    n = len(p)
    fitnesses = np.zeros((n))
    for k, ind in zip(range(n), p):
        fitnesses[k] = ind.fitness.values[0]
    return(fitnesses)


def _nparray_to_ind(ind_class, x):
    return(ind_class(x))


def _nparray_to_pop(f, x):
    return [f(x[:, i]) for i in range(x.shape[1])]


def _evaluate(eval_, p):
    invalid_ind = [ind for ind in p if not ind.fitness.valid]
    e = eval_(_pop_to_nparray(invalid_ind))
    for ind, fit in zip(invalid_ind, np.transpose(e)):
        ind.fitness.values = fit

    return(len(invalid_ind))


# -----------------------------------------------------------------------------
# Simple genetic algorithm
# -----------------------------------------------------------------------------
class Simple:
    def __init__(self, pop, eval_, verbose=0, CXPB=0.5, MUTPB=0.2):
        self.__eval = eval_
        self.__CXPB = CXPB
        self.__MUTPB = MUTPB
        self.__verbose = verbose

        # output
        if self.__verbose:
            print("--- Simple ---")

        # setup toolbox
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.__toolbox = base.Toolbox()
        self.__toolbox.register("individual", _nparray_to_ind, creator.Individual)
        self.__toolbox.register("population", _nparray_to_pop, self.__toolbox.individual)
        self.__toolbox.register("mate", tools.cxOnePoint)
        self.__toolbox.register("select", tools.selTournament, tournsize=10)
        self.__toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        self.__toolbox.register("evaluate", _evaluate, self.__eval)

        # setup statistics
        self.__stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.__stats.register("avg", np.mean)
        self.__stats.register("std", np.std)
        self.__stats.register("min", np.min)
        self.__stats.register("max", np.max)

        # initialize and evaluate population
        self.__pop = self.__toolbox.population(x=pop)
        self.__neval = self.__toolbox.evaluate(self.__pop)

        # output
        self._print_stats(0)

    def _print_stats(self, n):
        if self.__verbose:
            r = self.__stats.compile(self.__pop)
            print("%.2d: max=%.4f, min=%.4f, avg=%.4f, std=%.4f, neval=%d" %
                  (n, r['max'], r['min'], r['avg'], r['std'], self.__neval))

    def iter(self, n=100):
        # run generations
        if n > 0:
            for g in range(0, n):
                self.__pop = self.__toolbox.select(self.__pop, len(self.__pop))
                offspring = varAnd(self.__pop, self.__toolbox, self.__CXPB, self.__MUTPB)
                self.__neval += self.__toolbox.evaluate(offspring)
                self.__pop = offspring
                self._print_stats(g + 1)

        if n < 0:
            n = -n
            n_equal = 0
            v_last = None
            g = 1
            while(True):
                self.__pop = self.__toolbox.select(self.__pop, len(self.__pop))
                offspring = varAnd(self.__pop, self.__toolbox, self.__CXPB, self.__MUTPB)
                self.__neval += self.__toolbox.evaluate(offspring)
                self.__pop = offspring
                self._print_stats(g)
                g += 1

                # check if n equals have been found
                r = self.__stats.compile(self.__pop)
                v = r['min']
                if v == v_last:
                    n_equal += 1
                else:
                    n_equal = 0
                v_last = v
                if n_equal == n - 1:
                    break

        return([self.pop(), self.fitness()])

    def pop(self):
        return(_pop_to_nparray(self.__pop))

    def fitness(self):
        return(_popfitness_to_nparray(self.__pop))
