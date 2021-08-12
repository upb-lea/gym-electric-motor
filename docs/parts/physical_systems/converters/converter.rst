Power Electronic Converters
##################################

..  toctree::
    :maxdepth: 1
    :caption: Available Converters:

    1QC
    2QC
    4QC
    B6C
    DoubleConv
    NoConv

The converters are divided into two classes: The discretely controlled and continuously controlled converters.
The PowerElectronicConverter class is the base class for all converters. From that, the DiscreteConverter and the
ContinuousDynamicallyAveragedConverter derive to be the base class for all continuous and discrete converters.


Converter Base Class
'''''''''''''''''''''''''''''''

.. autoclass:: gym_electric_motor.physical_systems.converters.PowerElectronicConverter
   :members:

Finite Control Set Converter
'''''''''''''''''''''''''''''''

.. autoclass:: gym_electric_motor.physical_systems.converters.FiniteConverter
   :members:
   :inherited-members:


Continuous Control Set Dynamically Averaged Converter
'''''''''''''''''''''''''''''''''''''''''''''''''''''

.. autoclass:: gym_electric_motor.physical_systems.converters.ContDynamicallyAveragedConverter
   :members:
   :inherited-members:
