'''
This script does grid search. 
The runner module determines log dir names, spawns process in the most vacant 
GPU, and saves stdout/stderr if you wish. All you need is to define the 
hyperparam space.
'''
from experiments.master import runner
import os
import logging
import sys

logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG, 
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

slave_dir = os.path.dirname(os.path.abspath(__file__)) 
param_specs = {
    ('x', 'x1'): [0.5, 0.6, -0.3],  # x and x1 will have same value
    'y': ['foo', 'bar!']
}
tasks = runner.list_tasks(
    'python3 slave.py',
    param_specs,
    slave_dir,
    '/tmp/slave_logs/prefix')


r = runner.Runner(n_max_gpus=1, n_multiplex=4)
r.run_tasks(tasks)

