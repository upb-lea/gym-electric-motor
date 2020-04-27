Supply Converter Motor Load System (SCML)
#########################################

The technical structure of the SCML-Systems in the GEM-toolbox can bee seen in the figure below.

.. figure:: ../../plots/SCML_Setting.svg

The system consists of a Voltage Supply, a Power Electronic Converter, an Electrical Motor and the Mechanical Load.
Additionally, each SCML-System has got an ODE-Solver for the simulation.

..  toctree::
    :maxdepth: 1
    :caption: Subcomponents of the SCML:

    converters/converter
    mechanical_loads/mechanical_load
    noise_generators/noise_generator
    electric_motors/electric_motor
    voltage_supplies/voltage_supply
    ode_solvers/ode_solver



The abstract SCML-System defines the overall structure. From this, the DcMotorSystem and the SynchronousMotorSystem
derive. They only implement private methods. Therefore, the interface to the user stays the same in all cases.

.. autoclass:: gym_electric_motor.physical_systems.physical_systems.SCMLSystem
   :members:

Dc Motor System
****************
.. autoclass:: gym_electric_motor.physical_systems.physical_systems.DcMotorSystem
   :members:

Synchronous Motor System
************************
.. autoclass:: gym_electric_motor.physical_systems.physical_systems.SynchronousMotorSystem
   :members:

Squirrel Cage Induction Motor System
************************
.. autoclass:: gym_electric_motor.physical_systems.physical_systems.SquirrelCageInductionMotorSystem
   :members:

Doubly Fed Induction Motor System
************************
.. autoclass:: gym_electric_motor.physical_systems.physical_systems.DoublyFedInductionMotorSystem
   :members: