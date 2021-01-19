Environments
############

On this page, all environments with their environment-id are listed.
In general, all environment-ids are structured as follows:

``ControlType-ControlTask-MotorType-v0``

- The ``ControlType`` is in ``\{Disc / Cont\}``
- The ``ControlTask`` is in ``\{TC / SC / CC\}`` (Torque / Speed Current Control)
- The ``MotorType`` is in ``\{PermExDc / ExtExDc / SeriesDc / ShuntDc / PMSM / SynRM / DFIM / SCIM \}``



=================================================================== ==============================
Environment                                                          environment-id
=================================================================== ==============================
**Permanently Excited DC Motor Environments**

Discrete Torque Control Permanently Excited DC Motor Environment     ``'Disc-TC-PermExDc-v0'``
Continuous Torque Control Permanently Excited DC Motor Environment   ``'Cont-TC-PermExDc-v0'``
Discrete Speed Control Permanently Excited DC Motor Environment      ``'Disc-SC-PermExDc-v0'``
Continuous Speed Control Permanently Excited DC Motor Environment    ``'Cont-SC-PermExDc-v0'``
Discrete Current Control Permanently Excited DC Motor Environment    ``'Disc-CC-PermExDc-v0'``
Continuous Current Control Permanently Excited DC Motor Environment  ``'Cont-CC-PermExDc-v0'``

**Externally Excited DC Motor Environments**

Discrete Torque Control Externally Excited DC Motor Environment      ``'Disc-TC-ExtExDc-v0'``
Continuous Torque Control Externally Excited DC Motor Environment    ``'Cont-TC-ExtExDc-v0'``
Discrete Speed Control Externally Excited DC Motor Environment       ``'Disc-SC-ExtExDc-v0'``
Continuous Speed Control Externally Excited DC Motor Environment     ``'Cont-SC-ExtExDc-v0'``
Discrete Current Control Externally Excited DC Motor Environment     ``'Disc-CC-ExtExDc-v0'``
Continuous Current Control Externally Excited DC Motor Environment   ``'Cont-CC-ExtExDc-v0'``

**Series DC Motor Environments**

Discrete Torque Control Series DC Motor Environment                  ``'Disc-TC-SeriesDc-v0'``
Discrete Torque Control Series DC Motor Environment                  ``'Cont-TC-SeriesDc-v0'``
Discrete Speed Control  Series DC Motor Environment                  ``'Disc-SC-SeriesDc-v0'``
Continuous Speed Control Series DC Motor Environment                 ``'Cont-SC-SeriesDc-v0'``
Discrete Current Control Series DC Motor Environment                 ``'Disc-CC-SeriesDc-v0'``
Continuous Current Control Series DC Motor Environment               ``'Cont-CC-SeriesDc-v0'``

**Shunt DC Motor Environments**

Discrete Torque Control Shunt DC Motor Environment                   ``'Disc-TC-ShuntDc-v0'``
Continuous Torque Control Shunt DC Motor Environment                 ``'Cont-TC-ShuntDc-v0'``
Discrete Speed Control Shunt DC Motor Environment                    ``'Disc-SC-ShuntDc-v0'``
Continuous Speed Control Shunt DC Motor Environment                  ``'Cont-SC-ShuntDc-v0'``
Discrete Current Control Shunt DC Motor Environment                  ``'Disc-CC-ShuntDc-v0'``
Continuous Current Control Shunt DC Motor Environment                ``'Cont-CC-ShuntDc-v0'``

Continuous Dc Series Motor Environment                               ``'DcSeriesCont-v1'``
Discrete Dc Series Motor Environment                                  ``'DcSeriesDisc-v1'``

Continuous Dc Shunt Motor Environment                                 ``'DcShuntCont-v1'``
Discrete Dc Shunt Motor Environment                                   ``'DcShuntDisc-v1'``

Continuous Dc Permanently Excited Motor Environment                   ``'DcPermExCont-v1'``
Discrete Dc Permanently Excited Motor Environment                     ``'DcPermExDisc-v1'``

Continuous Dc Externally Excited Motor Environment                    ``'DcExtExCont-v1'``
Discrete Dc Externally Excited Motor Environment                      ``'DcExtExDisc-v1'``

Continuous Permanent Magnet Synchronous Motor Environment             ``'PMSMCont-v1'``
Discrete Permanent Magnet Synchronous Motor Environment               ``'PMSMDisc-v1'``

Continuous Synchronous Reluctance Motor Environment                   ``'SynRMCont-v1'``
Discrete Synchronous Reluctance Motor Environment                     ``'SynRMDisc-v1'``

Continuous Squirrel Cage Induction Motor Environment                  ``'SCIMCont-v1'``
Discrete Squirrel Cage Induction Motor Environment                    ``'SCIMDisc-v1'``

Continuous Doubly Fed Induction Motor Environment                     ``'DFIMCont-v1'``
Discrete Doubly Fed Induction Motor Environment                       ``'DFIMDisc-v1'``

=================================================================== ==============================

.. toctree::
   :maxdepth: 1
   :caption: Motor Environments:

   disc_tc_permex
   cont_tc_permex
   disc_cc_permex
   cont_cc_permex
   disc_sc_permex
   cont_sc_permex
   dc_extex_cont
   dc_extex_disc
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
