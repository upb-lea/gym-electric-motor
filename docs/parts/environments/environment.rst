Environments
############

On this page, all environments with their environment-id are listed.
In general, all environment-ids are structured as follows:

``ControlType-ControlTask-MotorType-v0``

- The ``ControlType`` is in ``{Finite / Cont}`` for all DC Motors and in ``{Finite / AbcCont / DqCont}`` for all AC Motors
- The ``ControlTask`` is in ``{TC / SC / CC}`` (Torque / Speed / Current Control)
- The ``MotorType`` is in ``{PermExDc / ExtExDc / SeriesDc / ShuntDc / PMSM / SynRM / DFIM / SCIM }``


=================================================================== ==============================
Environment                                                         environment-id
=================================================================== ==============================
**Permanently Excited DC Motor Environments**

Discrete Torque Control Permanently Excited DC Motor Environment     ``'Finite-TC-PermExDc-v0'``
Continuous Torque Control Permanently Excited DC Motor Environment   ``'Cont-TC-PermExDc-v0'``
Discrete Speed Control Permanently Excited DC Motor Environment      ``'Finite-SC-PermExDc-v0'``
Continuous Speed Control Permanently Excited DC Motor Environment    ``'Cont-SC-PermExDc-v0'``
Discrete Current Control Permanently Excited DC Motor Environment    ``'Finite-CC-PermExDc-v0'``
Continuous Current Control Permanently Excited DC Motor Environment  ``'Cont-CC-PermExDc-v0'``

**Externally Excited DC Motor Environments**

Discrete Torque Control Externally Excited DC Motor Environment      ``'Finite-TC-ExtExDc-v0'``
Continuous Torque Control Externally Excited DC Motor Environment    ``'Cont-TC-ExtExDc-v0'``
Discrete Speed Control Externally Excited DC Motor Environment       ``'Finite-SC-ExtExDc-v0'``
Continuous Speed Control Externally Excited DC Motor Environment     ``'Cont-SC-ExtExDc-v0'``
Discrete Current Control Externally Excited DC Motor Environment     ``'Finite-CC-ExtExDc-v0'``
Continuous Current Control Externally Excited DC Motor Environment   ``'Cont-CC-ExtExDc-v0'``

**Series DC Motor Environments**

Discrete Torque Control Series DC Motor Environment                  ``'Finite-TC-SeriesDc-v0'``
Discrete Torque Control Series DC Motor Environment                  ``'Cont-TC-SeriesDc-v0'``
Discrete Speed Control  Series DC Motor Environment                  ``'Finite-SC-SeriesDc-v0'``
Continuous Speed Control Series DC Motor Environment                 ``'Cont-SC-SeriesDc-v0'``
Discrete Current Control Series DC Motor Environment                 ``'Finite-CC-SeriesDc-v0'``
Continuous Current Control Series DC Motor Environment               ``'Cont-CC-SeriesDc-v0'``

**Shunt DC Motor Environments**

Discrete Torque Control Shunt DC Motor Environment                   ``'Finite-TC-ShuntDc-v0'``
Continuous Torque Control Shunt DC Motor Environment                 ``'Cont-TC-ShuntDc-v0'``
Discrete Speed Control Shunt DC Motor Environment                    ``'Finite-SC-ShuntDc-v0'``
Continuous Speed Control Shunt DC Motor Environment                  ``'Cont-SC-ShuntDc-v0'``
Discrete Current Control Shunt DC Motor Environment                  ``'Finite-CC-ShuntDc-v0'``
Continuous Current Control Shunt DC Motor Environment                ``'Cont-CC-ShuntDc-v0'``

**Permanent Magnet Synchronous Motor (PMSM) Environments**

Finite Torque Control PMSM Environment                               ``'Finite-TC-PMSM-v0'``
Abc-Continuous Torque Control PMSM Environment                       ``'AbcCont-TC-PMSM-v0'``
Dq-Continuous Torque Control PMSM Environment                        ``'DqCont-TC-PMSM-v0'``
Finite Speed Control PMSM Environment                                ``'Finite-SC-PMSM-v0'``
Abc-Continuous Speed Control PMSM Environment                        ``'Abc-Cont-SC-PMSM-v0'``
Dq-Continuous Speed Control PMSM Environment                         ``'Dq-Cont-SC-PMSM-v0'``
Finite Current Control PMSM Environment                              ``'Finite-CC-PMSM-v0'``
Abc-Continuous Current Control PMSM Environment                      ``'AbcCont-CC-PMSM-v0'``
Dq-Continuous Current Control PMSM Environment                       ``'DqCont-CC-PMSM-v0'``

**Synchronous Reluctance Motor (SynRM) Environments**

Finite Torque Control SynRM Environment                              ``'Finite-TC-SynRM-v0'``
Abc-Continuous Torque Control SynRM Environment                      ``'AbcCont-TC-SynRM-v0'``
Dq-Continuous Torque Control SynRM Environment                       ``'DqCont-TC-SynRM-v0'``
Finite Speed Control SynRM Environment                               ``'Finite-SC-SynRM-v0'``
Abc-Continuous Speed Control SynRM Environment                       ``'Abc-Cont-SC-SynRM-v0'``
Dq-Continuous Speed Control SynRM Environment                        ``'Dq-Cont-SC-SynRM-v0'``
Finite Current Control SynRM Environment                             ``'Finite-CC-SynRM-v0'``
Abc-Continuous Current Control SynRM Environment                     ``'AbcCont-CC-SynRM-v0'``
Dq-Continuous Current Control SynRM Environment                      ``'DqCont-CC-SynRM-v0'``

**Squirrel Cage Induction Motor (SCIM) Environments**

Finite Torque Control SCIM Environment                               ``'Finite-TC-SCIM-v0'``
Abc-Continuous Torque Control SCIM Environment                       ``'AbcCont-TC-SCIM-v0'``
Dq-Continuous Torque Control SCIM Environment                        ``'DqCont-TC-SCIM-v0'``
Finite Speed Control SCIM Environment                                ``'Finite-SC-SCIM-v0'``
Abc-Continuous Speed Control SCIM Environment                        ``'Abc-Cont-SC-SCIM-v0'``
Dq-Continuous Speed Control SCIM Environment                         ``'Dq-Cont-SC-SCIM-v0'``
Finite Current Control SCIM Environment                              ``'Finite-CC-SCIM-v0'``
Abc-Continuous Current Control SCIM Environment                      ``'AbcCont-CC-SCIM-v0'``
Dq-Continuous Current Control SCIM Environment                       ``'DqCont-CC-SCIM-v0'``

**Doubly Fed Induction Motor (DFIM) Environments**

Finite Torque Control DFIM Environment                               ``'Finite-TC-DFIM-v0'``
Abc-Continuous Torque Control DFIM Environment                       ``'AbcCont-TC-DFIM-v0'``
Dq-Continuous Torque Control DFIM Environment                        ``'DqCont-TC-DFIM-v0'``
Finite Speed Control DFIM Environment                                ``'Finite-SC-DFIM-v0'``
Abc-Continuous Speed Control DFIM Environment                        ``'Abc-Cont-SC-DFIM-v0'``
Dq-Continuous Speed Control DFIM Environment                         ``'Dq-Cont-SC-DFIM-v0'``
Finite Current Control DFIM Environment                              ``'Finite-CC-DFIM-v0'``
Abc-Continuous Current Control DFIM Environment                      ``'AbcCont-CC-DFIM-v0'``
Dq-Continuous Current Control DFIM Environment                       ``'DqCont-CC-DFIM-v0'``
=================================================================== ==============================

.. toctree::
   :maxdepth: 1
   :caption: Motor Environments:

   finite_tc_permex
   cont_tc_permex
   finite_cc_permex
   cont_cc_permex
   finite_sc_permex
   cont_sc_permex

   finite_tc_extex
   cont_tc_extex
   finite_cc_extex
   cont_cc_extex
   finite_sc_extex
   cont_sc_extex

   finite_tc_series
   cont_tc_series
   finite_cc_series
   cont_cc_series
   finite_sc_series
   cont_sc_series

   finite_tc_shunt
   cont_tc_shunt
   finite_cc_shunt
   cont_cc_shunt
   finite_sc_shunt
   cont_sc_shunt

   finite_tc_pmsm
   abccont_tc_pmsm
   dqcont_tc_pmsm
   finite_cc_pmsm
   abccont_cc_pmsm
   dqcont_cc_pmsm
   finite_sc_pmsm
   abccont_sc_pmsm
   dqcont_sc_pmsm

   finite_tc_synrm
   abccont_tc_synrm
   dqcont_tc_synrm
   finite_cc_synrm
   abccont_cc_synrm
   dqcont_cc_synrm
   finite_sc_synrm
   abccont_sc_synrm
   dqcont_sc_synrm

   finite_tc_scim
   abccont_tc_scim
   dqcont_tc_scim
   finite_cc_scim
   abccont_cc_scim
   dqcont_cc_scim
   finite_sc_scim
   abccont_sc_scim
   dqcont_sc_scim

   finite_tc_dfim
   abccont_tc_dfim
   dqcont_tc_dfim
   finite_cc_dfim
   abccont_cc_dfim
   dqcont_cc_dfim
   finite_sc_dfim
   abccont_sc_dfim
   dqcont_sc_dfim


Electric Motor Base Environment
'''''''''''''''''''''''''''''''

.. automodule:: gym_electric_motor.core

.. figure:: ../../plots/TopLevelStructure.svg

.. autoclass:: gym_electric_motor.core.ElectricMotorEnvironment
   :members:
