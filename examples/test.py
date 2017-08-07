from experiments import utils

parser = utils.parser('test')
parser.add_argument('-x', default=0.5, type=float)

if __name__ == '__main__':
    args = parser.parse_args()
    utils.preflight(args)
