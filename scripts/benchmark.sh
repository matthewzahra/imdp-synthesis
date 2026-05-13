#!/usr/bin/env bash

# reset any previous benchmarks
rm -rf output/*
rm -rf RL/agent_envs/*
rm -rf RL/agents/*
rm -rf sac_logs

# run MountainCar - includes sphere scaling
python RunFile.py --model MountainCar --rl --scale_sphere 0 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 0.25 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 0.5 --no_train --use_stamp 
python RunFile.py --model MountainCar --rl --scale_sphere 0.75 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 1 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 1.25 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 1.5 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 1.75 --no_train --use_stamp
python RunFile.py --model MountainCar --rl --scale_sphere 2 --no_train --use_stamp

# run Pendulum
python RunFile.py --model Pendulum --rl --no_train

# run Drone2D
python RunFile.py --model Drone2D --rl --no_train

# run Dubins_small
python RunFile.py --model Dubins_small --rl

# check different constant spheres in Dubins_small
python RunFile.py --model Dubins_small --rl --test_spheres --benchmark

# test constant spheres for Dubins_small
python RunFile.py --model Dubins_small --rl --constant_spheres