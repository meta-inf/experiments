import sys
import tensorflow as tf

from experiments import ipy_embed


__all__ = ['RealSupervisor', 'DebugSupervisor', 'create_sv']


class RealSupervisor():

    def __init__(self, logdir, save_model_secs=60, save_summaries_secs=120,
                 global_step=None):
        self.sv_ = tf.train.Supervisor(
            logdir=logdir, summary_op=None, 
            global_step=global_step,
            save_model_secs=save_model_secs,
            save_summaries_secs=save_summaries_secs)
        
    def __enter__(self):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.ctx_ = self.sv_.managed_session(config=tf_config)
        self.sess_ = self.ctx_.__enter__()
        return self

    def __exit__(self, *args, **kw):
        self.ctx_.__exit__(*args, **kw)

    def log(self, **kw):
        for tag in kw:
            if tag[0] == '_':
                continue
            s = tf.Summary()
            tag_path = tag.replace("_", "/")
            s.value.add(tag=tag_path, simple_value=float(kw[tag]))
            self.sv_.summary_computed(self.sess_, s)
        DebugSupervisor.log(self, **kw)


class DebugSupervisor():

    def __init__(self, extra_stack_depth=0, **kw):
        self.stack_depth = extra_stack_depth

    def __enter__(self):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess_ = tf.InteractiveSession(config=tf_config)
        self.sess_.run(tf.global_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return
        sys.stderr.write("{}\n{}\n{}\n".format(exc_type, exc_value, traceback))
        ipy_embed.embed(stack_depth=self.stack_depth)

    def log(self, **kw):
        if '_every' in kw and kw['_epoch'] % kw['_every'] != 0:
            return
        log_list = ["{} = {:.3f}".format(k, float(kw[k])) for k in kw if k[0] != '_']
        print("Epoch {}: {}".format(kw['_epoch'], ', '.join(log_list)))


def create_sv(args, **kw):
    if args.production:
        return RealSupervisor(args.dir, **kw)
    return DebugSupervisor(**kw)

