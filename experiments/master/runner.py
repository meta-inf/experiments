import os
import queue
import logging
import threading
import time
import re
import subprocess
from collections import namedtuple
from enum import Enum

from experiments.master import utils


Task = namedtuple('Task', 'cmd work_dir log_dir post_cmd group_id')
Task.__doc__ = """\
A task to run.
:param str cmd: command to run, preferrably excluding output redirection
:param str work_dir: working directory. 
:param str log_dir: directory to dump stdout/stderr
:param str post_cmd: command to run if task succeeds
:param str group_id: experiment group the task belongs to
"""


def run_task(t: Task):
    if t.log_dir is not None:
        # utils.preflight does sanity check
        os.makedirs(t.log_dir, exist_ok=True)
        fout = open(os.path.join(t.log_dir, 'stdout'), 'w')
        ferr = open(os.path.join(t.log_dir, 'stderr'), 'w')
    proc = subprocess.Popen(
        t.cmd, stdout=fout, stderr=ferr, cwd=t.work_dir, shell=True)
    proc.wait()
    fout.close(); ferr.close()
    return proc.returncode


class RunnerThread(threading.Thread):

    def __init__(self, env_str, runner):
        super(RunnerThread, self).__init__()
        self.env_setup = env_str
        self.runner = runner
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def should_stop(self):
        return self._stop_event.is_set()

    def run(self):
        while not self.should_stop():
            # Get task
            try:
                task = self.runner.que_todo.get(timeout=0.1)
            except queue.Empty as e:
                self.runner.logger.debug('Task queue empty. See you')
                time.sleep(2)
                continue
            # Run
            try:
                task = task._replace(cmd=self.env_setup + '  ' + task.cmd)
                id_ = utils.task_id(task)
                self.runner.logger.info(
                    'Launching task {}: {}\t'.format(id_, task.cmd))
                ret = run_task(task)
            except (IOError, subprocess.SubprocessError) as e:
                self.runner.logger.warn(
                    'task error: {}, {}'.format(id_, str(e)))
                ret = -100
            if ret != 0:
                self.runner.logger.info(
                    'task crashed: {}, {}'.format(id_, str(ret)))
                self.runner.que_failed.put(task)
            else:
                self.runner.logger.info('task completed: {}'.format(id_))
                self.runner.que_completed.put(task)
            self.runner.que_todo.task_done()


class Runner:

    def __init__(self, n_max_gpus, n_multiplex):
        self.que_todo = queue.Queue()
        self.que_failed = queue.Queue()
        self.que_completed = queue.Queue()
        self.gpus = utils.get_devices(n_max_gpus) * n_multiplex
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.threads = []
        for g in self.gpus:
            env_str = 'CUDA_VISIBLE_DEVICES={}'.format(g)
            th = RunnerThread(env_str, self)
            th.start()
            self.threads.append(th)
        
    def run_tasks(self, tasks):
        n_tasks = len(tasks)
        for t in tasks:
            self.que_todo.put(t)
        while True:
            if self.que_completed.qsize() + self.que_failed.qsize() >= n_tasks:
                if self.que_failed.empty():
                    self.logger.info('All tasks succeeded. Exiting')
                    for th in self.threads:
                        th.stop()
                    return
                self.logger.info('Restarting the following tasks: ')
                while not self.que_failed.empty():
                    t = self.que_failed.get()
                    self.logger.info('> ' + t.cmd)
                    self.que_todo.put(t)


def list_tasks(root_cmd, spec_list, work_dir, log_dir,
               post_cmd=None,
               group_id=None):
    '''
    Helper function to generate task list
    '''
    if type(spec_list) == dict:
        spec_list = list(spec_list.items())
    if spec_list == []:
        return [Task(
            cmd=root_cmd + ' -dir={}'.format(log_dir), 
            work_dir=work_dir, log_dir=log_dir, 
            post_cmd=post_cmd, group_id=group_id)]
    param, values = spec_list[0]
    ret = []
    used_names = set()
    for v in values:
        # Rename the logdir
        if len(values) > 1:
            new_name = utils.safe_path_str('{}_{}'.format(param[:2], v))
            if new_name in used_names:
                new_name += '_'; i = 0
                while new_name + str(i) in used_names:
                    i += 1
                new_name = new_name + str(i)
            used_names.add(new_name)
            new_log_dir = log_dir + '_' + new_name
        else:
            new_log_dir = log_dir
        # 
        ret += list_tasks(
                root_cmd + ' -{}={}'.format(param, v), 
                spec_list[1:], work_dir, new_log_dir, 
                post_cmd, group_id)
    return ret
