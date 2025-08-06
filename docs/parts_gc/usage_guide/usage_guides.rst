Usage Guides
_________________


Create a GEM environment
########################

To create a GEM environment use the gem.make() command. First of all, the type of the environment is defined by
the env_id. This specifies the action space (`'Finite'` or `'Cont'`), the control task ( `'SC'` for speed control,
`'TC'` for torque control or `'CC'` for current control) and the motor type. There are many parameters to customize
the environment. In the following code block, an example for the initialization of a speed control of an PMSM is
shown. A detailed description for the configuration of a GEM environment can be found in the
`GEM Cookbook <https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/environment_features/GEM_cookbook.ipynb>`_.

.. code-block:: python

   import numpy as np
   import gym_electric_motor as gem
   from gym_electric_motor.physical_systems import ConstantSpeedLoad
   from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
   from gym_electric_motor.visualization import MotorDashboard

   env_id = 'Cont-SC-PMSM-v0'

   # Define the motor parameters
   motor_parameters = dict(p=3, r_s=17.932e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.65e-3)

   # Define the nominal and limit values of the states
   nominal_values = dict(omega=4000*2*np.pi/60, i=230, u=350)
   limit_values = {key: 1.5 * nomin for key, nomin in nominal_values.items()}

   # Define the mechanical load
   load = ConstantSpeedLoad(load_initializer={'random_init': 'uniform', 'interval':[100,200] })

   # Define the reference generator
   rg = WienerProcessReferenceGenerator(reference_state='torque', sigma_range=(1e-3, 1e-2))

   # Define the states to be plotted
   visualization = MotorDashboard(state_plots=('torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq'))


   env = gem.make(
                env_id,
                tau=1e-5,   # Sampling time
                visualization=visualization,
                motor=dict(
                    motor_parameter=motor_parameter,
                    limit_values=limit_values,
                    nominal_values=nominal_values,
                ),
                load=load,
                reference_generator=rg,
   )

Create a GEM controller
#######################

To create a GEM Controller the `GemControllers.make()` command is used. There are some keyword arguments to adjust the controller.
For a detailed description see the `GEM Control Cookbook <https://colab.research.google.com/github/upb-lea/gem-control/blob/sphinx_doc/examples/GEM_Control_Cookbook.ipynb>`_.

* **env**
      The GEM environment that should be controlled.

* **env_id**
      The id of the GEM environment, that includes the action space, the control task and the motor type.

* **decoupling**
      Boolean to indicate, if an EMF-Feedforward corretion stage should be used.


* **current_safety_margin**
      Float to define a maximum current for the operation point selection.

* **base_speed_controller**
      Selection of the basic control algorithm for the speed control stage. Available control algorithms are 'PI', 'PID', 'P', 'ThreePoint'.

* **base_current_controller**
      Selection of the basic control algorithm for the current control stage. Available control algorithms are 'PI', 'PID', 'P', 'ThreePoint'.

* **a**
      Tuning parameter of the symmetrical optimum, that has to be larger than 1. A small value leads to a fast response, but also to a high overshoot.

* **plot_references**
      Boolean to indicate, if the reference values of the subordinated stages should be plotted. Therefore the referenced states of the subordinated stages have to be plotted.

* **block_diagram**
      Boolean to indicate, if a block diagram of the controller should be generated.

* **save_block_diagram_as**
      Selection of the data type for saving the block diagram. Also a tuple of data types can be passed. The available data types are 'pdf' and 'tex'. A window to select the folder and the file name will be opened.

.. code-block:: python

   import gem_controllers as gc

   c = gc.GemController.make(env=env,
                             env_id=env_id,
                             decoupling=True,
                             current_safety_margin=0.25,
                             base_speed_controller='PI',
                             base_current_controller='PI',
                             a=7,
                             plot_reference=True,
                             block_diagram=True,
                             save_block_diagram_as=('pdf',),
   )


Control a GEM environment
#########################

To use a GEM controller for controlling a GEM environment, the function c.control_environment() is available. If block_diagram
is `True`, a window will open, that shows the block diagram. There are some arguments, that could be passed.

* **env**
      The GEM environment to be controlled. This should be the same environment, that was used for the make command of the controller

* **n_steps**
      The number of iteration steps.

* **max_episode_length**
      The maximum number of steps, before the environment and the controller are reset.

* **render_env**
      Boolean to indicate, if the states of the environment should be plotted.

.. code-block:: python

   c.control_environment(env=env,
                         n_steps=10000,
                         max_episode_length=1000,
                         render_env=True)