Environments
############

In this part, all environments are explained. For each motor, an environment exists for the continuous and the discrete
action type. Functions that are the same for each DC motor are summarized in the dc_base_environment.
The PMSM environments are different due to its three phases and further motor states.
Nevertheless, some parts, as the reference generation, are similar to the DC motors.

.. toctree::
   :maxdepth: 2
   :caption: DC Motor Environments:

   dc_base_environment
   dc_extex_environment
   dc_series_environment
   dc_shunt_environment
   dc_permex_environment


.. toctree::
   :maxdepth: 2
   :caption: Permanently Magnet Synchronous Motor:

   pmsm_environment
