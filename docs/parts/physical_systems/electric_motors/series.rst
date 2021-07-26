DC Series Motor
############################

Schematic
*********

.. figure:: ../../../plots/ESBseries.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i}{\mathrm{d} t} &= \frac{u - L_\mathrm{e}^\prime i \omega_\mathrm{me} - (R_\mathrm{a} + R_\mathrm{e}) i}{L_\mathrm{a} + L_\mathrm{e}} \\


Torque Equation
***************
.. math::
    T = L_\mathrm{e}^\prime i^2

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.DcSeriesMotor
   :members:
   :inherited-members:
