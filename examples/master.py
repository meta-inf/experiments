from experiments.master import runner
import os
import logging
import sys

logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG, 
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

slave_dir = os.path.dirname(os.path.abspath(__file__)) 
param_specs = {'x': [0.5, 0.6, -0.3], 'y': ['foo', 'bar!']}
tasks = runner.list_tasks(
        'python3 slave.py -production',
        param_specs,
        slave_dir,
        '/tmp/slave_logs/prefix')


r = runner.Runner(n_max_gpus=1, n_multiplex=4)
r.run_tasks(tasks)

