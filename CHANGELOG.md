# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
