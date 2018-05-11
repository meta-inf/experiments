from __future__ import print_function
import os
import sys
import shutil
import argparse
import subprocess
import inspect


def parser(file_name):
    '''
    Add arguments needed by the framework
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', default='/tmp/{}/last'.format(file_name), type=str)
    parser.add_argument('-resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('-production', dest='production', action='store_true')
    parser.set_defaults(production=False)
    return parser


def source_dir():
    path = os.path.abspath(inspect.getmodule(inspect.stack()[1][0]).__file__)
    return os.path.dirname(path)


def preflight(args, data_dump=None, create_logdir=True):
    '''
    Routine checks, backup parameters 
    :param create_logdir: whether this worker should create args.dir
    '''
    def get_output(cmd):
        cp = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.PIPE)
        return cp.decode('utf-8')

    # Get git info    
    if args.production:
        # Please keep working directory clean
        try:
            out = get_output('git status')
        except subprocess.CalledProcessError as e:
            raise Exception('Git check failed: {}'.format(str(e)))
        # if str(out).find('working directory clean') == -1:
        #     raise Exception('Working directory not clean.')
        # Get commit hash
        commit_hash = get_output('git rev-parse HEAD').rstrip()
    else:
        # Get commit and modifications if possible
        try:
            diff_to_head = get_output('git diff HEAD')
            commit_hash = get_output('git rev-parse HEAD').rstrip()
        except Exception as e:
            print('Cannot get repo info:', e, file=sys.stderr)
            commit_hash = 'None'
            diff_to_head = str(e)
    #
    print('Commit: {}; production: {}'.format(commit_hash[:8], args.production))

    args.dir = os.path.expanduser(args.dir)
    if create_logdir and not args.resume:
        # Check if checkpoints exists. 
        # As runner may creates args.dir in advance, check hps dump
        if os.path.exists(os.path.join(args.dir, 'hps.txt')):
            if args.production:
                raise Exception('Directory {} exists'.format(args.dir))
            else:
                shutil.rmtree(args.dir)
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)
    else:
        # They must
        assert os.path.exists(args.dir)

    # Dump hyperparameters and other stuff
    with open(os.path.join(args.dir, 'hps.txt'), 'w') as fout:
        import json
        dct = args.__dict__
        dct['commit_hash'] = commit_hash
        print(json.dumps(dct), file=fout)

    if not args.production:
        with open(os.path.join(args.dir, 'repo-diff.txt'), 'w') as fout:
            print(diff_to_head, file=fout)

    with open(os.path.join(args.dir, 'dat.bin'), 'wb') as fout:
        import pickle
        pickle.dump(data_dump, fout)
