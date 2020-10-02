#!/usr/bin/env python
# coding=utf-8

"""User script to conduct the first hypothesis in the course"""


import logging
import itertools

import numpy

numpy.seterr(divide="ignore")

from . import database, preprocessor, algorithm, analysis


def infer_one(protocol, variables):
    """Runs one single test, returns the CER on the test set"""

    # 1. get the data from our preset API for the database
    train = database.get(protocol, "train", database.CLASSES, variables)

    # 2. preprocess the data using our module preprocessor
    norm = preprocessor.estimate_norm(numpy.vstack(train))
    train_normed = preprocessor.normalize(train, norm)

    # 3. trains our logistic regression system
    trainer = algorithm.MultiClassTrainer()
    machine = trainer.train(train_normed)

    # 4. applies the machine to predict on the 'unseen' test data
    test = database.get(protocol, "test", database.CLASSES, variables)
    test_normed = preprocessor.normalize(test, norm)
    test_predictions = machine.predict(numpy.vstack(test_normed))
    test_labels = algorithm.make_labels(test).astype(int)
    return analysis.CER(test_predictions, test_labels)


def infer_impact_of_variables_single(tabnum):
    """Builds the first table of my report"""

    for n, p in enumerate(database.PROTOCOLS):

        print(
            "\nTable %d: Single variables for Protocol `%s`:" % (n + tabnum, p)
        )
        print(60 * "-")

        for k in database.VARIABLES:
            result = infer_one(p, [k])
            print(("%-15s" % k), "| %d%%" % (100 * result,))


def infer_impact_of_variables_2by2(tabnum):
    """Builds the first table of my report"""

    for n, p in enumerate(database.PROTOCOLS):

        print(
            "\nTable %d: Variable combinations, 2x2 for Protocol `%s`:"
            % (n + tabnum, p)
        )
        print(60 * "-")

        for k in itertools.combinations(database.VARIABLES, 2):
            result = infer_one(p, k)
            print(("%-30s" % " + ".join(k)), "| %d%%" % (100 * result,))


def infer_impact_of_variables_3by3(tabnum):
    """Builds the first table of my report"""

    for n, p in enumerate(database.PROTOCOLS):

        print(
            "\nTable %d: Variable combinations, 3x3 for Protocol `%s`:"
            % (n + tabnum, p)
        )
        print(60 * "-")

        for k in itertools.combinations(database.VARIABLES, 3):
            result = infer_one(p, k)
            print(("%-45s" % " + ".join(k)), "| %d%%" % (100 * result,))


def infer_impact_of_variables_all(tabnum):
    """Builds the first table of my report"""

    for k, p in enumerate(database.PROTOCOLS):

        print("\nTable %d: All variables for Protocol `%s`:" % (k + tabnum, p))
        print(60 * "-")

        result = infer_one(p, database.VARIABLES)
        print(
            ("%-45s" % " + ".join(database.VARIABLES)),
            "| %d%%" % (100 * result,),
        )
