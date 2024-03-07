# GEM Control

[**Overview**](#Overview)
| [**Getting Started**](#Getting-Started)
| [**Installation**](#Installation)
| [**Citing**](#Citing)

[Read the paper (IEEE Xplore)](https://ieeexplore.ieee.org/document/10239044)

## Overview
This repository will contain controllers for the [gym-electric-motor](https://upb-lea.github.io/gym-electric-motor/) environments.
The following motors are supported:

- DC Motors:
    - Permanently Excited DC Motor
    - Externally Excited DC Motor
    - Series DC Motor
    - Shunt DC Motor
    
- Three Phase Motors:
    - Permanent Magnet Synchronous Motor
    - Synchronous Reluctance Motor
    - Externally Exited Synchronous Motor
    - Squirrel Cage Induction Motor

## Getting Started

An easy way to get started with GEM Control is by playing around with the following interactive notebooks in Google
Colaboratory. Most important features of GEM Control as well as application demonstrations are showcased, and give a kickstart
for engineers in industry and academia.

   - [GEM Control cookbook](https://colab.research.google.com/github/upb-lea/gem-control/blob/main/examples/GEM_Control_Cookbook.ipynb)
   - [Example Script](https://github.com/upb-lea/gem-control/blob/sphinx_doc/examples/example.py)

A basic routine is as simple as:
```py
import gym_electric_motor as gem
import gem_controllers as gc

env_id = 'Cont-TC-PMSM-v0'
env = gem.make(env_id)
c = gc.GemController.make(env, env_id)

c.control_environment(env, n_steps=10001, render_env=True)
```

## Installation

- Install from GitHub source:

```
git clone git@github.com:upb-lea/gem-control.git 
cd gem-control
# Then either
python setup.py install
# or alternatively
pip install -e .

# or alternatively
pip install git+https://github.com/upb-lea/gem-control
```

## Citing
Please cite the corresponding whitepaper when using this package:
```
@INPROCEEDINGS{10.1109/IEMDC55163.2023.10239044,
  author={Book, Felix and Traue, Arne and Schenke, Maximilian and Haucke-Korber, Barnabas and Wallscheid, Oliver},
  booktitle={2023 IEEE International Electric Machines & Drives Conference (IEMDC)}, 
  title={Gym-Electric-Motor (GEM) Control: An Automated Open-Source Controller Design Suite for Drives}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IEMDC55163.2023.10239044}}
```
