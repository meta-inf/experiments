import tensorflow as tf
import numpy as np
import tqdm

from experiments.slave.tp_utils import *
from experiments.slave import logs


__all__ = ['Symbol', 'Average', 'ScalarAverage', 'Op', 'List', 'traverse_ds', 'LogContext']


class Symbol:

  def __init__(self, key, sym):
    self.key = key
    self.sym = sym

  def gather_fn(self, lst):
    raise NotImplemented()

  def log_to_stdout(self):
    return False


class Average(Symbol):
  def __init__(self, key, sym):
    super(Average, self).__init__(key, sym)

  def gather_fn(self, lst):
    return np.mean(np.array(lst), axis=0)


class ScalarAverage(Average):
  def __init__(self, key, sym, log_to_stdout=True):
    super(ScalarAverage, self).__init__(key, sym)
    self._log_to_stdout = log_to_stdout

  def gather_fn(self, lst):
    return np.mean(np.array(lst))

  def log_to_stdout(self):
    return self._log_to_stdout


class Op(Symbol):
  def __init__(self, key, sym):
    super(Op, self).__init__(key, sym)

  def gather_fn(self, lst):
    return None


class List(Symbol):
  def __init__(self, key, sym):
    super(List, self).__init__(key, sym)

  def gather_fn(self, lst):
    return lst


def traverse_ds(symbols, ds_iter, sess, desc=None, callback=None):

  accu = dict((s.key, []) for s in symbols)
  tf_syms = []

  assert isinstance(symbols, list)
  for sym in symbols:
    assert isinstance(sym, Symbol)
    tf_syms.append(sym.sym)

  with tqdm.tqdm(ds_iter, desc=desc) as tr:
    for j, feed_dict in enumerate(tr):
      values = sess.run(tf_syms, feed_dict)
      for s, v in zip(symbols, values):
        accu[s.key].append(v)
      if callback is not None:
        cb_kw = dict((s.key, v) for s, v in zip(symbols, values))
        cb_kw['__locals'] = locals()
        cb_kw['__fd'] = feed_dict
        callback(sess, j, **cb_kw)
      per_iter_stat = dict(
        (s.key, np.mean(accu[s.key])) for s in symbols if s.log_to_stdout())
      tr.set_postfix(**per_iter_stat)

  ret = {}
  for sym in symbols:
    r = sym.gather_fn(accu[sym.key])
    if r is not None:
      ret[sym.key] = r

  return ret


_log_context = None


class LogContext:

  def __init__(self, n_epochs, global_step=None, tfsummary=False, logdir=None,
               max_queue=10, flush_secs=30):
    self._n_epochs = n_epochs
    self._global_step = global_step
    self._tfsummary = tfsummary
    if logdir is None and tfsummary:
      logdir = logs.get_logdir()
    self._logdir = logdir
    self._max_queue = max_queue
    self._flush_secs = flush_secs

  def __enter__(self):
    global _log_context
    assert _log_context is None, "Don't nest LogContexts"
    _log_context = self
    self._tq_obj = tqdm.trange(self._n_epochs)
    self._trange = self._tq_obj.__enter__()

    if self._tfsummary:
      self._tf_writer = tf.summary.FileWriter(
        self._logdir, graph=tf.get_default_graph(),
        max_queue=self._max_queue, flush_secs=self._flush_secs)

    return self

  def __exit__(self, *args):
    self._tq_obj.__exit__(args)
    global _log_context
    _log_context = None
    if self._tfsummary:
      self._tf_writer.close()

  def __iter__(self):
    for self._cur_ep in self._trange:
      yield self._cur_ep
      if self._tfsummary:
        self._tf_writer.flush()

  def log_scalars(self, val_dict, keys):
    vd = {}
    for k in keys:
      vd[k] = val_dict[k]
      assert len(np.shape(vd[k])) == 0, "{} must be scalar ({})".format(k, np.shape(vd[k]))
    self._trange.set_postfix(**vd)

    if self._tfsummary:
      gs = self._global_step.eval() if self._global_step is not None else self._cur_ep
      for k in keys:
        s = create_scalar_summary(k, vd[k])
        self._tf_writer.add_summary(s, gs)

  def log_image(self, key, image):
    if self._tfsummary:
      gs = self._global_step.eval() if self._global_step is not None else self._cur_ep
      s = create_image_summary(key, image_to_nhwc(image))
      self._tf_writer.add_summary(s, gs)
