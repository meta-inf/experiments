'''
This is your experiment script. Apart from having the experiments module 
installed, you only need to ensure it resides in a git repo.
'''
from experiments import utils
import time
import sys
import numpy as np


'''
utils.parser returns an argparse.ArgumentParser, with the following fields set:
- dir: directory to hold hyperparameter dump and repo snapshot. 
       You can put your tensorflow logs here as well.
- resume: if training was resumed. `utils.preflight` use this field to 
       determine if you need cleaning or sanity check, to avoid mess up by
       supervisor's default restore behavior.
- production: if strict sanity check will be enforced.
'''
parser = utils.parser('script_name')
parser.add_argument('-x', default=0.5, type=float)
parser.add_argument('-x1', default=0.5, type=float)
parser.add_argument('-y', default='bla', type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    '''
    preflight dumps hyperparameters (and additional data if needed), 
    store the commit hash / diff patch of the repo, 
    and does cleaning and sanity check
    '''
    utils.preflight(args, data_dump=None)
    assert abs(args.x - args.x1) < 1e-3
    # Proceed to your experiment
    if args.x < 0:
        raise Exception('Hey! Don\'t do that!')
    time_ = 0.5 / (1 + np.exp(-args.x))
    time.sleep(time_)
    print(time_)
    print('err: ', time_, file=sys.stderr)

