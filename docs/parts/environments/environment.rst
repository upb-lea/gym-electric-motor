Environments
############

On this page, all environments with their environment-id are listed.
Furthermore, a short introduction into the Interface of the environments is given.

========================================================= ==============================
Environment                                                   environment-id
========================================================= ==============================
Continuous Dc Series Motor Environment                    ``'DcSeriesCont-v1'``
Discrete Dc Series Motor Environment                      ``'DcSeriesDisc-v1'``

Continuous Dc Shunt Motor Environment                     ``'DcShuntCont-v1'``
Discrete Dc Shunt Motor Environment                       ``'DcShuntDisc-v1'``

Continuous Dc Permanently Excited Motor Environment       ``'DcPermExCont-v1'``
Discrete Dc Permanently Excited Motor Environment         ``'DcPermExDisc-v1'``

Continuous Dc Externally Excited Motor Environment        ``'DcExtExCont-v1'``
Discrete Dc Externally Excited Motor Environment          ``'DcExtExDisc-v1'``

Continuous Permanent Magnet Synchronous Motor Environment ``'PMSMCont-v1'``
Discrete Permanent Magnet Synchronous Motor Environment   ``'PMSMDisc-v1'``

Continuous Synchronous Reluctance Motor Environment       ``'SynRMCont-v1'``
Discrete Synchronous Reluctance Motor Environment         ``'SynRMDisc-v1'``
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


Electric Motor Base Environment
'''''''''''''''''''''''''''''''

.. automodule:: gym_electric_motor.core

.. figure:: ../../plots/TopLevelStructure.svg

.. autoclass:: gym_electric_motor.core.ElectricMotorEnvironment
   :members:
