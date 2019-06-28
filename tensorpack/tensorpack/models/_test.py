#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: _test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
import unittest

class TestModel(unittest.TestCase):
    def run_variable(self, var):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        if isinstance(var, list):
            return sess.run(var)
        else:
            return sess.run([var])[0]

    def make_variable(self, *args):
        if len(args) > 1:
            return [tf.Variable(k) for k in args]
        else:
            return tf.Variable(args[0])

def run_test_case(case):
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    import tensorpack
    from tensorpack.utils import logger
    from . import *
    logger.disable_logger()
    subs = tensorpack.models._test.TestModel.__subclasses__()
    for cls in subs:
        run_test_case(cls)


