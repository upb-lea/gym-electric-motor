GEM Control API Documentation

.. toctree::
   :maxdepth: 1
   :caption: Subcomponents of a GEM Controller:

   stages/stages
   pi_current_controller
   torque_controller
   pi_speed_controller
   gem_adapter
   reference_plotter
   utils

.. autoclass:: gem_controllers.gem_controller.GemController
   :members: make, control, reset, tune, control_environment, signals, signal_names, get_signal_value, stages