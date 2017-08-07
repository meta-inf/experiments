import os
import sys
import shutil
import argparse
import subprocess


def add_args(parser: argparse.ArgumentParser, file_name):
    '''
    Add arguments needed by the framework
    '''
    parser.add_argument('-dir', default='/tmp/{}/last'.format(file_name), type=str)
    parser.add_argument('-resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('-production', dest='production', action='store_true')
    parser.set_defaults(production=False)


def preflight(args, data_dump=None):
    '''
    Routine checks, backup parameters 
    '''
    if args.production:
        # Check if working directory is clean
        try:
            out = subprocess.check_output('git status', shell=True)
        except subprocess.CalledProcessError as e:
            raise Exception('Git check failed: {}'.format(str(e)))
        if str(out).find('working directory clean') == -1:
            raise Exception('Working directory not clean.')
        # Get commit hash
        commit_hash = subprocess.check_output('git rev-parse HEAD', shell=True)
        commit_hash = commit_hash.decode('utf-8')
        print(commit_hash)
    else:
        try:
            commit_hash = subprocess.check_output('git rev-parse HEAD', shell=True)
            commit_hash = commit_hash.decode('utf-8')
            print(commit_hash)
        except Exception as e:
            print(e)
            commit_hash = str(e)
    # Make logdir
    if not args.resume and os.path.exists(args.dir):
        if args.production:
            raise Exception('Directory {} exists'.format(args.dir))
        shutil.rmtree(args.dir)
    os.makedirs(args.dir, exist_ok=True)
    # Dump hyperparameters and other stuff
    with open(os.path.join(args.dir, 'hps.txt'), 'w') as fout:
        import json
        dct = args.__dict__
        dct['commit_hash'] = commit_hash
        print(json.dumps(dct), file=fout)
    with open(os.path.join(args.dir, 'dat.bin'), 'wb') as fout:
        import pickle
        pickle.dump(data_dump, fout)
