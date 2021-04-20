---
title: 'gym-electric-motor (GEM): A Python toolbox for the simulation of electric drive systems'
tags:
  - Python
  - electric drive control
  - electric motors
  - OpenAI Gym
  - power electronics
  - reinforcement learning
authors:
  - name: Praneeth Balakrishna
    affiliation: 1
    
  - name: Gerrit Book
    affiliation: 1
    
  - name: Wilhelm Kirchgässner
    orcid: 0000-0001-9490-1843
    affiliation: 1
    
  - name: Maximilian Schenke
    orcid: 0000-0001-5427-9527
    affiliation: 1
    
  - name: Arne Traue
    affiliation: 1
    
  - name: Oliver Wallscheid
    orcid: 0000-0001-9362-8777
    affiliation: 1
    
affiliations:
 - name: Department of Power Electronics and Electrical Drives, Paderborn University, Germany
   index: 1
date: 28. May 2020
bibliography: Literature.bib
---

# Summary

The ``gym-electric-motor`` (``GEM``) library provides simulation environments for electrical drive systems and, therefore, allows to easily design and analyze drive control solutions in Python.
Since ``GEM`` is strongly inspired by OpenAI's ``gym`` [@gym-whitepaper], it is particularly well-equipped for (but not limited to) applications in the field of reinforcement-learning-based control algorithms. 
In addition, the interface allows to plug in any expert-driven control approach, such as model predictive control, to be tested  and to perform benchmark comparisons. 
The ``GEM`` package includes a wide variety of motors, power electronic converters and mechanical load models that can be flexibly selected and parameterized via the API. 
A modular structure allows additional system components to be included in the simulation framework.

# Statement of Need

Electric drive systems and their control are an important topic in both academic and industrial research due to their worldwide usage and deployment. 
Control algorithms for these systems have usually been designed, parameterized and tested within ``MATLAB - Simulink`` [@MathWorks], which is developed and promoted specifically for
such engineering tasks. 
In the more recent past, however, commercial software like ``MATLAB`` has difficulties to stay on par with state-of-the-art concepts for scientific modeling and the flexibility offered by open-source libraries that are available for more accessible programming languages like Python. 
Consequently, a Python-based drive simulation framework like ``GEM`` is an evident step in order to accelerate corresponding control research and development.
Specifically, the latest efforts concerning industrial application of reinforcement-learning control algorithms heavily depend on Python packages like ``Keras`` [@Chollet2015], ``Tensorflow`` [@tensorflow2015-whitepaper] or ``PyTorch``[@NEURIPS2019_9015]. 
Hence, the built-in OpenAI ``gym`` interface allows to easily couple ``GEM`` to other open-source reinforcement learning toolboxes such as ``Stable Baselines3`` [@stable-baselines3], ``TF-Agents`` [@TFAgents] or ``keras-rl`` [@plappert2016kerasrl].

Providing easy access to the non-commercial, open-source ``GEM`` library allows users from any engineering domain to include accurate drive models into their simulations, also beyond the topic of control applications.
Considering the prevalence of commercial software like ``MATLAB`` for educational purposes, a free-of-charge simulation alternative that does not force students or institutions to pay for licenses, has great potential to support and encourage training of new talents in the field of electrical drives and neighbouring domains (e.g. power electronics or energy systems).
``GEM`` has already been used in graduate courses on reinforcement learning [@rl-lecture].

# Related software

Due to the strong dependence of downstream industrial development on simulated environments there is a comprehensive variety of commercial software that enables numerical analysis of every facet of electric drives. 
To name just a few, ``MATLAB - Simulink`` is probably the most popular software environment for numerical analysis in engineering.
Herein, ``MATLAB`` is providing for a scientific calculation framework and ``Simulink`` for a model-driven graphical interface with a very large field of applications. 
Examples that are designed for real-time capability (e.g., for hardware-in-the-loop prototyping) can be found in ``VEOS`` [@dSPACE] or ``HYPERSIM`` [@OPAL-RT].
Non-commercial simulation libraries exist, but they rarely come with predefined system models. 
An exemplary package from this category is ``SimuPy`` [@Margolis], which provides lots of flexibility for the synthesis of generic simulation models, but also requires the user to possess the necessary expert knowledge in order to implement a desired system model. 
Likewise, general purpose component-oriented simulation frameworks like ``OpenModelica`` [@OSMC2020] or ``XCos`` [@Scilab2020] can be used for setting up electrical drive models, too, but this requires expert domain knowledge and out-of-the-box Python interfaces (e.g., for reinforcement learning) are not available. 

In the domain of motor construction it is furthermore interesting to observe the behavior of magnetic and electric fields within a motor (simulation of partial differential equations).
Corresponding commercial simulation environments, like ``ANSYS Maxwell`` [@ANSYS], ``Motor-CAD`` [@MotorDesignLtd] or ``MotorWizard`` [@ElectroMagneticWorks] and the exemplary non-commercial alternative ``FEMM`` [@Meeker] are very resource and time consuming because they depend on the finite element method, which is a spatial discretization and numerical integration procedure. 
Hence, these software packages are usually not considered in control development, and complement ``GEM`` at most. 
This particularly applies in the early control design phase when researching new, innovative control approaches (rapid control prototyping) or when students want to receive quasi-instantaneous simulation feedbacks. 

# Package Architecture

The ``GEM`` library models an electric drive system by its four main components: voltage supply, power converter, electric motor and mechanical load. 
The general structure of such a system is depicted in \autoref{fig:SCML_system}. 

![Simplified structure diagram of an electric drive system\label{fig:SCML_system}](../plots/SCML_Setting.eps)

The __voltage supply__ provides the necessary power that is used by the motor. 
It is modeled by a fixed supply voltage $u_\mathrm{sup}$, which allows to monitor the supply current into the converter.
A __power electronic converter__ is needed to supply the motor with electric power of proper frequency and magnitude, which commonly includes the conversion of the supply's direct current to alternating current. 
Typical drive converters exhibit switching behavior: there is a finite set of different voltages that can be applied to the motor, depending on which switches are open and which are closed. 
Besides this physically accurate view, a popular modeling approach for switched mode converters is based on dynamic averaging of the applied voltage $u_\mathrm{in}$, rendering the voltage a continuous variable.
Both of these modeling approaches are implemented and can be chosen freely, allowing usage of control algorithms that operate on a finite set of switching states or on continuous input voltages.
The __electric motor__ is the centerpiece of every drive system. 
It is described by a system of ordinary differential equations (ODEs), which represents the motor's electrical behavior. 
In particular, the domain of three-phase drives makes use of coordinate transformations to view these ODEs in the more interpretable frame of field-oriented coordinates. 
In ``GEM``, both, the physically accurate three-phase system ($abc$-coordinates) and the simplified, two-dimensional, field-oriented system ($dq$-coordinates) are available to be used as the frame of input and output variables, allowing for easy and quick controller analysis and diagnose within the most convenient coordinate system. 
Finally, the torque $T$ resulting from the motor is applied to the __mechanical load__. 
The load is characterized by a moment of inertia and by a load torque $T_\mathrm{L}$ that is directed against the motor torque. 
Load torque behavior can be parameterized with respect to the angular velocity $\omega_\mathrm{me}$ in the form of constant, linear and quadratic dependency (and arbitrary combinations thereof). 
Speed changes that result from the difference between motor and load torque are modeled with another ODE which completely covers the mechanical system behavior.
Alternatively, the motor speed can be set to a fixed value, which can be useful for the investigation of control algorithms concerning generator operation, or it can be set to follow a specified trajectory, which is convenient when inspecting scenarios with defined speed demands like in traction applications. 

# Features

A large number of different motor systems is already implemented. 
These include DC drives as well as synchronous and induction three-phase drives. 
A complete list can be viewed in the ``GEM`` documentation [@GEM-docu].
The corresponding power converters allow to control the motor either directly via applied voltage (continuous-control-set) or by defining the converter's switching state (finite-control-set). 
Specifically for the use within reinforcement-learning applications and for testing state-of-the-art expert-driven control designs, the toolbox comes with a built-in reference generator, which can be used to create arbitrary reference trajectories (e.g., for the motor current, velocity or torque). 
These generated references are furthermore used to calculate a reward. In the domain of reinforcement learning, reward is the optimization variable that is to be maximized. 
For the control system scenario, reward is usually defined by the negative distance between the momentary and desired operation point, such that expedient controller behavior can be monitored easily.
The reward mechanism also allows to take physical limitations of the drive system into account, e.g., in the way of a notably low reward if limit values are surpassed. 
Optionally, the environment can be setup such that a reset of the system is induced in case of a limit violation. 
In addition, built-in visualization and plotting routines allow to monitor the training process of reinforcement learning agents or the performance of expert-driven control approaches.

# Examples

A minimal example of ``GEM's`` simulation capability is presented in \autoref{fig:SCIM_example}.
The plot shows the start-up behavior of a squirrel cage induction motor connected to an idealized three-phase electric grid depicting the angular velocity $\omega_\mathrm{me}$, the torque $T$, the voltage $u_{a,b,c}$ and the current $i_{d,q}$.
Here, the voltage is depicted within the physical $abc$-frame while the current is viewed within the simplified $dq$-frame. 

![Simulation of a squirrel cage induction motor connected to a rigid network at $50 \, \mathrm{Hz}$\label{fig:SCIM_example}](../plots/SCIM_Example.eps)

Exemplary code snippets that demonstrate the usage of ``GEM`` within both, the classical control and the reinforcement learning context are included within the project's [examples folder](https://github.com/upb-lea/gym-electric-motor/tree/master/examples). 
Featured examples:

- [``GEM_cookbook.ipynb``](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/environment_features/GEM_cookbook.ipynb): a basic tutorial-style notebook that presents the basic interface and usage of GEM
- [``scim_ideal_grid_simulation.py``](https://github.com/upb-lea/gym-electric-motor/blob/master/examples/environment_features/scim_ideal_grid_simulation.py): a simple motor simulation showcase of the squirrel cage induction motor that was used to create \autoref{fig:SCIM_example}



# References
