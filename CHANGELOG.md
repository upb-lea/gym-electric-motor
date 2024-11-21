# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
## Changed
## Fixed

## [3.0.3] - unreleased
## Added
## Changed
- Changed minimal required gymnasium version to 0.29.1.
- updated the code of gem-control to be compatible with gymnasium v1.0.0
## Fixed
- #263 updated the sb3, mpc and gem-control examples to run with gymnasium v1.0.0 

## [3.0.2] - 2024-11-19
## Added
## Changed
## Fixed
- Fixed the packaging in the pyproject.toml to successful instal gym-electric-motor from the pypi database

## [3.0.0] - 2024-11-18
## Added
- Support for Python 3.11, 3.12, 3.13
- Ruff: Python linter & formatter (see [DEVELOPMENT.md](DEVELOPMENT.md))
- StateObserver: An easy way to get state values with error checking [example](examples/observers/state_observer_example.py)
- Integrated gem_controls repository into gem. classic_controllers will be removed in further version
- Using pyproject.toml, dropping deprecated setup.py
- Enabled Gymnasium env checker [see here](https://gymnasium.farama.org/api/experimental/wrappers/#gymnasium.experimental.wrappers.PassiveEnvCheckerV0)
## Changed
- Dropped support for Python 3.8
- Linted and formatted all files
- Changed max. steps in some test files to improve test speed by 30%
- Changed the syntax from gem_controller.py to be compatible with the gymnasium interface
## Fixed
- #244 Sphinx docu build
- #233 EESM ODE update
- #224 DqToAbcActionProcessor uses now the correct angle to apply the dq to abc transformation 
- #223 CosSinProcessor uses now the denormalized angel for calculating the sin and cos
- #253 #256 all examples to the current syntax

## [2.0.0] - 2023-08-15
## Added
- Support for Python 3.10
- Changed API to be compatible with Gymnasium v0.29.0
- Output a Motor Dashborad graph as PNG or PDF file (with LaTeX font)
- Automated integration test (run a motor simulation with controller)

## Changed
- Dropped support for Python 3.7 (due to gymnasium not supporting it)

## Fixed
- Bugfixes for EESM environment

## [1.1.0] - 2022-04-25
## Added
- Physical System Wrappers as a new feature to process actions and states directly in the environment. A decent introduction can be found in the [GEM cookbook](https://github.com/upb-lea/gym-electric-motor/blob/nightly/examples/environment_features/GEM_cookbook.ipynb) (Paragraph 2.8)
- The externally excited synchronous motor (EESM) has been added to the GEM-toolbox. 
- The environments of the EESM can be instantiated with the following keys: "{Cont|Finite}-{CC|TC|SC}-EESM-v0",

## Changed
- The MotorDashboard has received a "initialize()" method to initialize the plots below a specific cell.
- The MotorDashboard is now compatible with the "%matplotlib widget" backend. Therefore, GEM is now compatible with the integrated jupiter notebook execution of Visual Studio Code

## Fixed
- If multiple converters were used and the time constant tau was changed from its default values, it was possible that the values of tau were different in each converter


## [1.0.1] - 2021-12-20
## Added
- Classic field oriented controllers for induction motors
- Uniform initialization of the WienerProcessReferenceGenerator

## Changed
- Reduced the dynamics of the reference signals in several speed control environments
- Changed the default ode-solver of all environments to the ScipyOdeSolver

## Fixed
- gym version compatibility for all versions >= 0.21.0
- Docs: m2r dependency to m2r2. Enables compatibility with latest sphinx versions.
- matplotlib compatibility with versions >= 3.5.0
- Bugfix in the stable_baselines3_dqn_disc_pmsm_example.ipynb example notebook
- Bugfix in the jacobian of the ConstantSpeedLoad

## [1.0.0] - 2021-08-23
### Added
- classic controllers in examples with complete makeover
- Possibility to seed all random number generators at once with a unified API - Reproduciblity improved.

### Changed
#### Environment IDs
- The environments have changed the IDs for their creation. 
- Furthermore, environments specialized for torque (TC), current (CC) and speed (SC) control tasks have been introduced.
- The ids are now structured as follows:
{_Cont/Finite_}-{_CC/TC/SC_}-_motorType_-v0
- Details can be looked up on the code documentation pages.
#### gem.make() parameters
The arbitrary environment creation keyword arguments that were passed to every subcomponent of the environment
have been removed. Instead, there is a fixed set of keyword parameters for every environment described in the 
API-documentation of every environment.

#### MPC example
- Visualization improvements
- fix: State variables were updated

#### Miscellaneous
- Documentation was updated
- Changed all DC Envs to FourQuadrantConverters per default
- Adjusted the dynamics of the speed references in DC environments
- Adjusted the plots for better visibility of single points


### Removed
- simple_controllers.py in examples
- Tensorforce tutorial due to deprecation


## [0.3.1] - 2020-12-18
### Added
- Constraint monitor class

### Changed
- Visualization framework
