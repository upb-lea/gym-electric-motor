Polynomial Static Load
######################

Mechanical ODE
''''''''''''''

.. math::
    \frac{ \mathrm{d} \omega_{me} } { \mathrm{d} t } = \frac{ T - T_L(\omega_{me})}{J_{total}}


Polynomial Load Equation
''''''''''''''''''''''''

.. math::
    T_L(\omega_{me})=\mathrm{sign}(\omega_{me})(c \omega^2_{me} + b \vert\omega_{me}\vert + a)\\

Class Description
''''''''''''''''''
.. autoclass:: gym_electric_motor.physical_systems.mechanical_loads.PolynomialStaticLoad
    :members:
    :inherited-members:
