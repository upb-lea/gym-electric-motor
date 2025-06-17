.. gym-electric-motor documentation master file, created by
   sphinx-quickstart on Tue Jul  2 15:49:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gym-electric-motor(GEM)'s documentation!
===================================================

The gym-electric-motor (GEM) package is a software toolbox for the simulation of different electric motors to
train and test reinforcement learning motor controllers and to compare them with classical motor controllers.


Getting started
***************************

A quick start guide can be found in the following Readme-File.

.. toctree::
   :maxdepth: 1
   :caption: Gym Electric Motor Readme:

   parts_gem/readme


Content
*******

In the environments section all available GEM-environments are presented with their default configuration.
For quick start, one of these can be selected and used out of the box.


The documentation of the base classes is important for the development of own modules like further reward functions or
reference generators. In this part, the basic interfaces of each module are specified.
For the development of physical models like further motor models or further mechanical load models, the physical system
documentation specifies the basic interfaces inside a physical system.

..  toctree::
    :maxdepth: 4
    :titlesonly:
    :caption: gym-electric-motor Contents:

    parts_gem/environments/environment
    parts_gem/reference_generators/reference_generator
    parts_gem/reward_functions/reward_function
    parts_gem/physical_systems/physical_system
    parts_gem/physical_system_wrappers/physical_system_wrapper
    parts_gem/visualizations/visualization
    parts_gem/constraint_monitor
    parts_gem/core
    parts_gem/utils
    parts_gem/callbacks
    parts_gem/random_component

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. gym-electric-motor documentation master file, created by
   sphinx-quickstart on Tue Jul  2 15:49:19 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GEM-controls documentation!
===================================================

The GEM-control package is a collection of control methods to control the gym-electric-motor(GEM) environments.
Currently, classic PI controllers are available for most of the environment types.

Contents
________

.. toctree::
   :maxdepth: 1
   :caption: Gym Electric Controller Contents:

   parts_gc/read_me
   parts_gc/usage_guide/usage_guides
   parts_gc/api_documentation/gem_control

