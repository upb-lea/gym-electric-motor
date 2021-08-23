Polynomial Static Load
######################

Mechanical ODE
''''''''''''''

.. math::
    \frac{ \mathrm{d} \omega_\mathrm{me} } { \mathrm{d} t } = \frac{ T - T_\mathrm{L} (\omega_\mathrm{me})}{J_\mathrm{total}}


Polynomial Load Equation
''''''''''''''''''''''''

.. math::
    T_\mathrm{L}(\omega_\mathrm{me})=\mathrm{sign}(\omega_\mathrm{me})(c \omega^2_\mathrm{me} + b \vert\omega_\mathrm{me}\vert + a)\\

Class Description
''''''''''''''''''
.. autoclass:: gym_electric_motor.physical_systems.mechanical_loads.PolynomialStaticLoad
    :members:
    :inherited-members:
