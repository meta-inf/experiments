import experiments.utils as utils


__all__ = ['get_logdir']


_log_dir = None


def get_logdir():
  global _log_dir
  if _log_dir is not None:
    return _log_dir

  if utils._log_dir is not None:
    _log_dir = utils._log_dir
    return _log_dir
  
  raise NotImplemented()
