#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File:         parallel.py
@Author:       Guangquan ZENG
@Contact:      guangquan.zeng@outlook.com
@Description:  Some parallel utilities.
'''


import os
from joblib import parallel_backend
from joblib import Parallel, delayed


def run(function, inputs, parallel=True, n_jobs=-1, backend='loky'):
    """
    >> output = function(input)
    # the "input" represents one or multiple positional arguments.

    >> outputs = parallel_function(function, inputs)
    # inputs = [input_0, input_1, ...]
    # outputs = [output_0, output_1, ...]
    """
    if len(inputs) == 0:
        raise ValueError("No inputs provided for parallel processing.")
    if parallel:
        if n_jobs == -1:
            n_jobs = min(len(inputs), os.cpu_count())
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            try:
                outputs = Parallel()(
                    delayed(function)(*input) for input in inputs)
            except TypeError:
                outputs = Parallel()(
                    delayed(function)(input) for input in inputs)
    else:
        try:
            outputs = [function(*input) for input in inputs]
        except TypeError:
            outputs = [function(input) for input in inputs]
    return outputs
