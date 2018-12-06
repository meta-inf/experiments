'''
This script does grid search. 
The runner module determines log dir names, spawns process in the most vacant 
GPU, and redirects stdout/stderr of the slave.
'''

from experiments.master import runner
import logging
import os
import shutil
import sys


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG, 
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

ctr = {
    0: 0,
    1: 0
}

def on_task_finish(status, task, code=-1):
    # Hook for gathering results. Check task.log_dir if you wish. Here we do a sanity check
    assert code != runner.Status.LAUNCH_FAILED
    assert (status == runner.Status.CRASHED) == (
        task.option_dict['x'] < 0 or task.option_dict['sleep_long'] is True)
    ctr[status] += 1

slave_working_dir = os.path.dirname(os.path.abspath(__file__))
param_specs = {
    ('x', 'x1'): [0.5, 0.6, -0.3],  # x and x1 will have same value
    'y': ['foo', 'bar!'],
    'sleep_long': runner.BooleanOpt()
}

log_dir = '/tmp/slave_logs/'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

tasks = runner.list_tasks(
    'python3 slave.py',
    param_specs,
    slave_working_dir,
    log_dir + 'prefix',
    max_cpu_time=2,
    post_kill_cmd='echo "KILLED"')

r = runner.Runner(
    n_max_gpus=1, n_multiplex=4, n_max_retry=-1, on_task_finish=on_task_finish)
r.run_tasks(tasks)

print(ctr)
