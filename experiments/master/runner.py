import os
import queue
import logging
import threading
import time
import re
import subprocess
import shutil
from collections import namedtuple
from enum import Enum

from experiments.master import utils


Task = namedtuple('Task', 'cmd option_dict work_dir log_dir post_cmd group_id ttl max_cpu_time post_kill_cmd')
Task.__doc__ = """\
A task to run.
:param str cmd: command to run, preferrably excluding output redirection
:param str work_dir: working directory. 
:param str log_dir: directory to dump stdout/stderr
:param str post_cmd: command to run if task succeeds
:param str group_id: experiment group the task belongs to
:param int ttl: restart counter
"""


def run_task(t: Task):
    if t.log_dir is not None:
        # utils.preflight does sanity check
        os.makedirs(t.log_dir, exist_ok=True)
        fout = open(os.path.join(t.log_dir, 'stdout'), 'w')
        ferr = open(os.path.join(t.log_dir, 'stderr'), 'w')
    proc = subprocess.Popen(
        t.cmd, stdout=fout, stderr=ferr, cwd=t.work_dir, shell=True)
    try:
        proc.communicate(timeout=t.max_cpu_time)
    except subprocess.TimeoutExpired:
        proc.kill()
        if len(t.post_kill_cmd.strip()) > 0:
            time.sleep(0.5)
            os.system(t.post_kill_cmd)
    fout.close(); ferr.close()
    return proc.returncode


class Status:
    SUCCEED = 0
    CRASHED = 1
    LAUNCH_FAILED = -1000


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
            task_raw = task
            try:
                task = task._replace(cmd=self.env_setup + '  ' + task.cmd)
                id_ = utils.task_id(task)
                self.runner.logger.info(
                    'Launching task {}: {}\t'.format(id_, task.cmd))
                ret = run_task(task)
            except (IOError, subprocess.SubprocessError) as e:
                self.runner.logger.warn(
                    'error launching task: {}, {}'.format(id_, str(e)))
                ret = -Status.LAUNCH_FAILED
            task = task_raw
            if ret != 0:
                self.runner.logger.warning(
                    'task crashed: {}, {}'.format(id_, str(ret)))
                self.runner.on_task_finish(Status.CRASHED, task, code=ret)
                if task.ttl > self.runner.max_ttl:
                    self.runner.logger.warning(
                        'task {} reaches maximum ttl. abandoned'.format(id_))
                    self.runner.finished_tasks.inc()
                else:
                    if task.log_dir and os.path.exists(task.log_dir):
                        # Move it so next run doesn't automatically fail
                        i = 0
                        while os.path.exists(task.log_dir + str(i)):
                            i += 1
                        shutil.move(task.log_dir, task.log_dir + str(i))
                    task = task._replace(ttl=task.ttl+1)
                    self.runner.que_failed.put(task)
            else:
                self.runner.logger.info('task completed: {}'.format(id_))
                self.runner.on_task_finish(Status.SUCCEED, task)
                self.runner.que_completed.put(task)
                self.runner.finished_tasks.inc()
            self.runner.que_todo.task_done()


def get_logger():
    logger = logging.getLogger()
    logging.addLevelName(
        logging.INFO,
        "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(
        logging.WARNING,
        "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(
        logging.ERROR,
        "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    return logger


class Runner:

    def __init__(self, n_max_gpus, n_multiplex, n_max_retry=3,
                 on_task_finish=None,
                 log_level=logging.INFO):
        self.max_ttl = n_max_retry
        if on_task_finish is None:
            self.on_task_finish = lambda *args, **kw: None
        else:
            self.on_task_finish = on_task_finish

        self.finished_tasks = utils.AtomicCounter()
        self.que_todo = queue.Queue()
        self.que_failed = queue.Queue()
        self.que_completed = queue.Queue()
        self.gpus = utils.get_devices(n_max_gpus) * n_multiplex
        self.logger = get_logger()
        self.logger.setLevel(log_level)
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
            if self.finished_tasks.value >= n_tasks:
                self.logger.info('All tasks finished. Exiting')
                for th in self.threads:
                    th.stop()
                return
            elif self.que_todo.empty(): 
                self.logger.info('Restarting the following tasks: ')
                while not self.que_failed.empty():
                    t = self.que_failed.get()
                    self.logger.info('- ' + t.cmd)
                    self.que_todo.put(t)
            time.sleep(2)


def _list_tasks(root_cmd, current_opt, remaining_opts, log_dir, task_params):
    if remaining_opts == []:
        task_params = task_params.copy()
        task_params['cmd'] = root_cmd + ' -dir={}'.format(log_dir)
        task_params['option_dict'] = current_opt
        task_params['log_dir'] = log_dir
        task_params['ttl'] = 0
        return [Task(**task_params)]

    param, values = remaining_opts[0]
    if type(param) == str:
        param = [param]
    else:
        assert type(param) == tuple
        param = list(param)
    ret = []
    used_names = set()
    for cval in values:
        # current spec: 'param[i]=cval[i]' for i in len(param)
        # when cval is atomic, it means all param in this item share the same 
        # value
        if type(cval) != list:
            cval = [cval] * len(param)
        else:
            assert len(cval) == len(param)
        # Rename the logdir: add -param[i]_cval[i] to the suffix
        if len(values) > 1:
            new_name = '-'.join(
                [utils.safe_path_str('{}_{}'.format(p_[:2], v_))
                 for p_, v_ in zip(param, cval)]
            )
            if new_name in used_names:
                new_name += '_'; i = 0
                while new_name + str(i) in used_names:
                    i += 1
                new_name = new_name + str(i)
            used_names.add(new_name)
            new_log_dir = log_dir + '_' + new_name
        else:
            new_log_dir = log_dir
        # Arg string
        new_args = ''
        new_opt = current_opt.copy()
        for p_, v_ in zip(param, cval):
            new_args += ' -{} {}'.format(p_, v_)
            new_opt[p_] = v_
        ret += _list_tasks(
            root_cmd + new_args, new_opt, remaining_opts[1:], new_log_dir, task_params)
    return ret


def list_tasks(root_cmd, spec_list, work_dir, log_dir,
               post_cmd=None, group_id=None, max_cpu_time=None,
               post_kill_cmd=None):
    '''
    Helper function to generate task list
    '''
    if type(spec_list) == dict:
        spec_list = list(spec_list.items())
    root_cmd += ' -production'    
    lc = locals()
    task_params = dict([(k, lc[k]) for k in [
        'work_dir', 'post_cmd', 'group_id', 'max_cpu_time', 'post_kill_cmd']])
    return _list_tasks(
            root_cmd, {}, spec_list, log_dir, task_params)

