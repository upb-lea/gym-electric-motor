Synchronous Reluctance Motor
############################

Schematic
*********

.. figure:: ../../../plots/ESBdqSynRM.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_{sd}}{\mathrm{d} t}&=\frac{u_{sd} + p \omega_{me} L_q i_{sq} - R_s i_{sd}}{L_d} \\
    \frac{\mathrm{d} i_{sq}}{\mathrm{d} t}&=\frac{u_{sq} - p \omega_{me} L_d i_{sd} - R_s i_{sq}}{L_q} \\
    \frac{\mathrm{d} \varepsilon_{el}}{\mathrm{d} t}&= p \omega_{me}

Torque Equation
***************
.. math::
    T = 1.5 p (L_d - L_q) i_{sd} i_{sq}


Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SynchronousReluctanceMotor
   :members:
   :inherited-members:
