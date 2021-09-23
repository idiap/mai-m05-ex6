#!/usr/bin/env python
# coding=utf-8

"""Test unit for analysis code"""

import numpy

from . import analysis, preprocessor, database, algorithm


def doit(predictions, true_labels, expected):
    """Runs a single test case

    Parameters
    ==========

    predictions : list
        A list of integer predictions to input

    true_labels : list
        Ground truth values to compare to

    expected : float
        The expected classification-error rate


    Raises
    ======

    AssertionError
        In case something goes wrong

    """

    predictions = numpy.array(predictions)
    true_labels = numpy.array(true_labels)

    cer = analysis.CER(predictions, true_labels)

    assert numpy.isclose(cer, expected), "Expected %r, but got %r" % (
        expected,
        cer,
    )


def test_CER_0():
    doit([0, 1], [0, 1], 0)


def test_CER_50_50():
    doit([1, 1], [0, 1], 0.5)


def test_CER_20_80():
    doit([1, 1, 0, 1, 1], [1, 1, 1, 1, 1], 0.2)


def test_CER_1():
    doit([1, 1], [0, 0], 1)
