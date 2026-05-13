import argparse
import datetime


def parse_arguments():
    '''
    Function to parse arguments provided

    :return: Object with all arguments
    '''

    # Options
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, perform additional checks to debug python")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for random number generators (Jax, Numpy)")
    parser.add_argument('--decimals', type=int, default=4,
                        help="Number of decimals to work with for storing probabilities")
    parser.add_argument('--pAbs_min', type=float, default=0.0001,
                        help="Minimum probability for absorbing states")

    parser.add_argument('--gpu', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run on GPU. Otherwise, run on CPU")
    parser.add_argument('--gpu_rvi', action=argparse.BooleanOptionalAction, default=False,
                        help="If true, run RVI on GPU. Otherwise, run on CPU")

    parser.add_argument('--model', type=str, default='Drone2D',
                        help="Benchmark model to run")
    parser.add_argument('--model_version', type=int, default=0,
                        help="Version of the model to use (optinal; 0 by default)")
    parser.add_argument('--checker', type=str, default='storm',
                        help="Model checker to use (prism or storm)")
    parser.add_argument('--prism_dir', type=str, default='~/Documents/Tools/prism/prism/bin/prism',
                        help="Directory where Prism is located")

    parser.add_argument('--mode', type=str, default='fori_loop',
                        help="Should be one of 'fori_loop', 'vmap', 'python'")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for functions vectorized with Jax")

    # Plotting options
    parser.add_argument('--plot_grid', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot unit grids in figures")
    parser.add_argument('--plot_title', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot titles in figures")
    parser.add_argument('--plot_ticks', action=argparse.BooleanOptionalAction, default=False,
                        help="If True, plot ticks in figures")
    
    parser.add_argument('--rl',action="store_true", default=False,
                        help="Use Reinforcement Learning Agent")
    
    # parser.add_argument('--alg', type=str, default='SAC')

    parser.add_argument('--no_train', action='store_true', default=False,
                        help="don't train new RL agents, just use existing ones")
    
    parser.add_argument('--test_spheres', action='store_true', default=False,
                        help='test spheres defined in RL.sphere_defs.py for the satisfaction probabilities that they produce')
    
    parser.add_argument('--scale_sphere', type=float, default=1, 
                        help='scale the sphere radii by some scalar value')
    
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help="part of the benchmark")
    
    parser.add_argument('--constant_spheres', action='store_true', default=False,
                        help="do not use dynamcially shaped spheres")
    
    parser.add_argument('--use_stamp', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Parse arguments
    args = parser.parse_args()

    return args
