# -*- coding: UTF-8 -*-
# File: naming.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

GLOBAL_STEP_OP_NAME = 'global_step'
GLOBAL_STEP_VAR_NAME = 'global_step:0'

# prefix of predict tower
PREDICT_TOWER = 'towerp'

# extra variables to summarize during training in a moving-average way
MOVING_SUMMARY_VARS_KEY = 'MOVING_SUMMARY_VARIABLES'

# placeholders for input variables
INPUT_VARS_KEY = 'INPUT_VARIABLES'

import tensorflow as tf
SUMMARY_BACKUP_KEYS = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]

# export all upper case variables
all_local_names = locals().keys()
__all__ = [x for x in all_local_names if x.isupper()]
