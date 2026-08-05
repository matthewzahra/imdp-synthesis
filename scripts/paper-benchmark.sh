#!/usr/bin/env bash

# reset any previous benchmarks
rm -rf output/*
rm -rf RL/agent_envs/*
rm -rf RL/agents/*
rm -rf sac_logs

# MountainCar
python RunFile.py --model MountainCar --rl --constant_spheres --scale_sphere 0
python RunFile.py --model MountainCar --rl --constant_spheres --scale_sphere 0.5
python RunFile.py --model MountainCar --rl --constant_spheres --scale_sphere 0.75
python RunFile.py --model MountainCar --rl --constant_spheres --scale_sphere 1

# Dubins_small coarse partition
python RunFile.py --model Dubins_small --rl --constant_spheres --scale_sphere 0
python RunFile.py --model Dubins_small --rl --constant_spheres --scale_sphere 1
python RunFile.py --model Dubins_small --rl --constant_spheres --scale_sphere 1.5
python RunFile.py --model Dubins_small --rl --constant_spheres --scale_sphere 2

# Dubins_small fine partition
python RunFile.py --model Dubins_small --rl --constant_spheres --finer_partition --scale_sphere 0
python RunFile.py --model Dubins_small --rl --constant_spheres --finer_partition --scale_sphere 1
python RunFile.py --model Dubins_small --rl --constant_spheres --finer_partition --scale_sphere 1.5
python RunFile.py --model Dubins_small --rl --constant_spheres --finer_partition --scale_sphere 2

# Dubins_small variable sphere
python RunFile.py --model Dubins_small --rl --finer_partition

# DoubleIntegrator
python RunFile.py --model DoubleIntegrator --rl --constant_spheres --scale_sphere 0
python RunFile.py --model DoubleIntegrator --rl --constant_spheres --scale_sphere 1
python RunFile.py --model DoubleIntegrator --rl --constant_spheres --scale_sphere 3
python RunFile.py --model DoubleIntegrator --rl --constant_spheres --scale_sphere 5