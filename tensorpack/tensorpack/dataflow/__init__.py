#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
import importlib
import os
import os.path

from . import dataset
from . import imgaug

def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]

__SKIP = ['dftools', 'dataset', 'imgaug']
for _, module_name, _ in walk_packages(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_') and \
        module_name not in __SKIP:
        _global_import(module_name)

