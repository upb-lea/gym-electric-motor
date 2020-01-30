Permanently Excited DC Motor
#############################

Schematic
*********

.. figure:: ../../../plots/ESBdcpermex.svg


Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i}{\mathrm{d} t} &= \frac{u - \mathit{\Psi}_e i - R_a i}{L_a} \\


Torque Equation
***************
.. math::
    T = \mathit{\Psi}_e i



Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.DcPermanentlyExcitedMotor
   :members:
   :inherited-members:
