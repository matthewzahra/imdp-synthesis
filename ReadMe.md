# Introduction
The write up for this work can be found/downloaded from [here](https://matthewzahra.com/project/Robust%20Policy%20Optimisation%20with%20RL). This write up corresponds to the branch `main`. Current work is taking place on the branch `RL` where we are trying to improve our process for paper submission.

# IMDP Synthesis

This repository implements abstraction-based methods for robust policy synthesis in discrete-time stochastic dynamical systems. The goal is to compute a policy that maximizes the probability of satisfying a specification, with a formal lower bound on that probability. The default specification is reach-avoid: reach a goal set while avoiding unsafe (critical) regions. Finite-horizon reach-avoid and finite-horizon safety are also supported (see [Specifications](#specifications)).

The continuous system is abstracted into a finite model, on which robust dynamic programming computes a policy and a lower bound on the satisfaction probability. Two abstractions are implemented, each with its own entrypoint:

- **IMDP** (`Main_IMDP.py`): an interval Markov decision process.
- **SVMDP** (`Main_SVMDP.py`): a set-valued Markov decision process.

By default, the abstraction is restricted to a tube of states and actions around rollouts of a reinforcement learning (RL) policy (PPO or SAC). Use `--dense` to abstract the full state space instead.

> **Note:** At present, the SVMDP path performs much better than the IMDP path, and it is the recommended default. All experiment scripts in this repository (`ICLR_*.py`) use `Main_SVMDP.py`.

The codebase implements and builds on methods from the following papers:

- Thom Badings and Alessandro Abate. "Probabilistic Alternating Simulations for Policy Synthesis in Uncertain Stochastic Dynamical Systems." In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 3919-3924. IEEE, 2025.
- Thom Badings, Licio Romao, Alessandro Abate, David Parker, Hasan A. Poonawala, Marielle Stoelinga, and Nils Jansen. "Robust Control for Dynamical Systems with Non-Gaussian Noise via Formal Abstractions." Journal of Artificial Intelligence Research, 76:341-391, 2023.

# Installation

The code can be installed by following the instructions below.

### 1. Create Python environment

The first step is to create the Python environment. We tested the code on Python 3.12.

We recommend using (mini)conda for setting up an environment. To create and activate the environment, run:

```
conda create -n dynabs-jax python=3.12
conda activate dynabs-jax
```

Install cddlib and GMP by [following the (OS-dependent) instructions here](https://pycddlib.readthedocs.io/en/latest/quickstart.html). For example, on MacOS, you can run:

```
brew install cddlib gmp
```

Then, install the dependencies within the conda environment:

```
pip install -r requirements.txt
```

Finally, install pycddlib:

```
pip install pycddlib
```

If installing pycddlib gives you an error similar to ```Cannot open include file: 'cddlib/setoper.h': No such file or directory```, then try
to [use this troubleshoot page.](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation)
On MacOS, the suggested fix is as follows:

```
env "CFLAGS=-I$(brew --prefix)/include -L$(brew --prefix)/lib" python -m pip install pycddlib
```

### 2. Install JAX

To install JAX with CUDA support via conda, run:

```
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
```

To instead install JAX without CUDA support, run:

```
pip install jax==0.8.0
```

> We have also tested running with JAX on METAL. However, performance on Apple Silicon chips currently seems better on an up-to-date version of JAX+JAXlib (running on CPU) than on JAX on METAL.

### 3. Install Storm and Stormpy

We use the probabilistic model checker Storm (via its Python bindings provided by Stormpy) to compute optimal policies on interval MDP (IMDP) abstractions.
Install the following

1. Install Storm 1.11.0 (Make sure to install the required dependencies first: https://www.stormchecker.org/documentation/obtain-storm/build.html).
2. Install Stormpy 1.11.0 within the conda environment (https://moves-rwth.github.io/stormpy/).

# Reproducing experiments

The experiments presented in the paper can be reproduced by running the following commands:

```
python RunFile.py --model Dubins --model_version 0
python RunFile.py --model Dubins --model_version 1
python RunFile.py --model Dubins --model_version 2
```

These commands run the Dubin's vehicle benchmark with no, 10%, and 20% parameter uncertainty, respectively.

A smaller version of the Dubin's benchmark is also available (this can be useful for faster debugging purposes):

```
python RunFile.py --model Dubins_small
```

Created figures will be stored in the `output/` folder. The runtimes and model sizes can be read from the terminal output.

## Other benchmarks

Currently, the following benchmarks are implemented and have been tested:

- Dubins
- Dubins_small
- Drone2D
- MountainCar
- Pendulum

To run one of these benchmarks, simply replace the `--model ...` argument above with the respective benchmark name.

## Debugging

In case you run into memory issues, place try changing the `--batch_size` argument, which is set to 100,000 by default. A reasonable setting might be 30,000 or even 10,000.

