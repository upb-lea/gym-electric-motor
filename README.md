# Gym Electric Motor
![](docs/plots/Motor_Logo.png)

[**Paper**](https://arxiv.org/abs/1910.09434)
| [**Quickstart**](#getting-started)
| [**Install guide**](#installation)
| [**Reference docs**](https://upb-lea.github.io/gym-electric-motor/)
| [**Release notes**](https://github.com/upb-lea/gym-electric-motor/releases)

[![Build Status](https://travis-ci.org/upb-lea/gym-electric-motor.svg?branch=master)](https://travis-ci.org/upb-lea/gym-electric-motor)
[![codecov](https://codecov.io/gh/upb-lea/gym-electric-motor/branch/master/graph/badge.svg)](https://codecov.io/gh/upb-lea/gym-electric-motor)
[![PyPI version shields.io](https://img.shields.io/pypi/v/gym-electric-motor.svg)](https://pypi.python.org/pypi/gym-electric-motor/)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/upb-lea/gym-electric-motor/blob/master/LICENSE)
[![DOI Zenodo](https://zenodo.org/badge/216583232.svg)](https://zenodo.org/badge/latestdoi/216583232)
[![DOI JOSS](https://joss.theoj.org/papers/10.21105/joss.02498/status.svg)](https://doi.org/10.21105/joss.02498)

## Overview
The gym-electric-motor (GEM) package is a Python toolbox for the simulation and control of various electric motors.

It is built upon [OpenAI Gym Environments](https://gym.openai.com/), and, therefore, can be used for both, classical control simulation and r[einforcement learning](https://github.com/upb-lea/reinforcement_learning_course_materials) experiments. It allows you to construct a typical drive train with the usual building blocks, i.e. supply voltages, converters, electric motors and load models, and obtain not only a closed-loop simulation of this physical structure, but also a rich interface for plugging in any decision making algorithm, from PI-controllers to [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) agents.

## Getting Started
An easy way to get started with GEM is by playing around with the following interactive notebooks in Google Colaboratory. Most important features of GEM as well as application demonstrations are showcased, and give a kickstart for engineers in industry and academia.

* [GEM cookbook](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master//examples/environment_features/GEM_cookbook.ipynb)
* [Keras-rl2 example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/keras_rl2_dqn_disc_pmsm_example.ipynb)
* [Stable-baselines3 example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/stable_baselines3_dqn_disc_pmsm_example.ipynb)
* [Tensorforce example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/tensorforce_dqn_disc_pmsm_example.ipynb)
* [MPC  example](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/model_predictive_controllers/gekko_mpc_cont_pmsm_example.ipynb)

There is a list of [standalone example scripts](examples/) as well for minimalistic demonstrations.

A basic routine is as simple as:
```py
import gym_electric_motor as gem

if __name__ == '__main__':
    env = gem.make("Finite-CC-PMSM-v0")  # instantiate a discretely controlled PMSM
    env.reset()
    for _ in range(1000):
        env.render()  # visualize environment
        (states, references), rewards, done, _ =\ 
        	env.step(env.action_space.sample())  # pick random control actions
    env.close()
```



## Installation
- Install gym-electric-motor from PyPI (recommended):

```
pip install gym-electric-motor
```

- Install from Github source:

```
git clone git@github.com:upb-lea/gym-electric-motor.git 
cd gym-electric-motor
# Then either
python setup.py install
# or alternatively
pip install -e .
```

## Building Blocks
A GEM environment consists of following building blocks:
- Physical structure:
   - Supply voltage
   - Converter
   - Electric motor
   - Load model
- Utility functions for reference generation, reward calculation and visualization
 
### Information Flow in a GEM Environment
![](docs/plots/SCML_Overview.png)

Among various DC-motor models, the following three-phase motors - together with their power electronic counterparts - are available:
- Permanent magnet synchronous motor (PMSM), 
- Synchronous reluctance motor (SynRM)
- Squirrel cage induction motor (SCIM)
- Doubly-fed induction motor (DFIM)

The converters can be driven by means of a duty cycle (continuous mode) or switching commands (discrete mode). 

### Citation
A whitepaper for this framework is available under (preprint: [arxiv.org/abs/1910.09434](https://arxiv.org/abs/1910.09434), full-paper: [IEEE-Xplore](https://ieeexplore.ieee.org/document/9241851)). Please use the following BibTeX entry for citing us:
```
@ARTICLE{9241851,  
author={A. {Traue} and G. {Book} and W. {Kirchgässner} and O. {Wallscheid}},  
journal={IEEE Transactions on Neural Networks and Learning Systems},   
title={Toward a Reinforcement Learning Environment Toolbox for Intelligent Electric Motor Control},   
year={2020},  volume={},  number={},  
pages={1-10},  
doi={10.1109/TNNLS.2020.3029573}}
```

### Running Unit Tests with Pytest
To run the unit tests ''pytest'' is required.
All tests can be found in the ''tests'' folder.
Execute pytest in the project's root folder:
```
>>> pytest
```
or with test coverage:
```
>>> pytest --cov=./
```
All tests shall pass.
