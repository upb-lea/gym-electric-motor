Environments
############

On this page, all environments with their environment-id are listed.
Furthermore, a short introduction into the Interface of the environments is given.

========================================================= ==============================
Environment                                                   environment-id
========================================================= ==============================
Continuous Dc Series Motor Environment                    ``'emotor-dc-series-cont-v1'``
Discrete Dc Series Motor Environment                      ``'emotor-dc-series-disc-v1'``

Continuous Dc Shunt Motor Environment                     ``'emotor-dc-shunt-cont-v1'``
Discrete Dc Shunt Motor Environment                       ``'emotor-dc-shunt-disc-v1'``

Continuous Dc Permanently Excited Motor Environment       ``'emotor-dc-permex-cont-v1'``
Discrete Dc Permanently Excited Motor Environment         ``'emotor-dc-permex-disc-v1'``

Continuous Dc Externally Excited Motor Environment        ``'emotor-dc-extex-cont-v1'``
Discrete Dc Externally Excited Motor Environment          ``'emotor-dc-extex-disc-v1'``

Continuous Permanent Magnet Synchronous Motor Environment ``'emotor-pmsm-cont-v1'``
Discrete Permanent Magnet Synchronous Motor Environment   ``'emotor-pmsm-disc-v1'``

Continuous Synchronous Reluctance Motor Environment       ``'emotor-symrm-cont-v1'``
Discrete Synchronous Reluctance Motor Environment         ``'emotor-synrm-disc-v1'``

Continuous Squirrel Cage Induction Motor Environment      ``'emotor-scim-cont-v1'``
Discrete Squirrel Cage Induction Motor Environment        ``'emotor-scim-disc-v1'``

Continuous Doubly Fed Induction Motor Environment         ``'emotor-dfim-cont-v1'``
Discrete Doubly Fed Induction Motor Environment           ``'emotor-dfim-disc-v1'``
========================================================= ==============================

.. toctree::
   :maxdepth: 1
   :caption: Motor Environments:

   dc_extex_cont
   dc_extex_disc
   dc_permex_cont
   dc_permex_disc
   dc_series_cont
   dc_series_disc
   dc_shunt_cont
   dc_shunt_disc
   pmsm_cont
   pmsm_disc
   synrm_cont
   synrm_disc
   scim_cont
   scim_disc
   dfim_cont
   dfim_disc


Electric Motor Base Environment
'''''''''''''''''''''''''''''''

.. automodule:: gym_electric_motor.core

.. figure:: ../../plots/TopLevelStructure.svg

.. autoclass:: gym_electric_motor.core.ElectricMotorEnvironment
   :members:
