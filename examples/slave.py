from experiments import utils
import time
import sys
import numpy as np


parser = utils.parser('test')
parser.add_argument('-x', default=0.5, type=float)
parser.add_argument('-y', default='bla', type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    utils.preflight(args)
    if args.x < 0:
        raise Exception('Hey! Don\'t do that!')
    time_ = 0.5 / (1 + np.exp(-args.x))
    time.sleep(time_)
    print(time_)
    print('err: ', time_, file=sys.stderr)

