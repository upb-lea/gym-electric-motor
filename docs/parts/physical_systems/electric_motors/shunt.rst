DC Shunt Motor
############################

Schematic
*********
.. figure:: ../../../plots/ESBshunt.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_a}{\mathrm{d} t} &= \frac{u - L_e^\prime \omega_{me} i_e - R_a i_a}{L_a} \\
    \frac{\mathrm{d} i_e}{\mathrm{d} t} &= \frac{u  - R_e i_e}{L_e}


Torque Equation
***************
.. math::
    T = L_e^\prime i_e i_a


Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.DcShuntMotor
   :members:
   :inherited-members:
