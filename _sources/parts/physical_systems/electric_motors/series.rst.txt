DC Series Motor
############################

Schematic
*********

.. figure:: ../../../plots/ESBseries.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i}{\mathrm{d} t} &= \frac{u - L_e^\prime \omega_{me} i - (R_a + R_e) i}{L_a + L_e} \\


Torque Equation
***************
.. math::
    T = L_e^\prime i^2

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.DcSeriesMotor
   :members:
   :inherited-members:
