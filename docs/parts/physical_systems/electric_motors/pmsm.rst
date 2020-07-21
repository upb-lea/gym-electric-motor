Permanent Magnet Synchronous Motor
##################################

Schematic
*********

.. figure:: ../../../plots/ESBdq.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_{sd}}{\mathrm{d} t}&=\frac{u_{sd} + p \omega_{me} L_q i_{sq} - R_s i_{sd}}{L_d} \\
    \frac{\mathrm{d} i_{sq}}{\mathrm{d} t}&=\frac{u_{sq} - p \omega_{me} (L_d i_{sd} + \mathit{\Psi}_p) - R_s i_{sq}}{L_q} \\
    \frac{\mathrm{d} \varepsilon_{el}}{\mathrm{d} t}&= p \omega_{me}



Torque Equation
***************

.. math:: T=\frac{3}{2} p (\mathit{\Psi}_p +(L_d-L_q)i_{sd}) i_{sq}

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.PermanentMagnetSynchronousMotor
   :members:
   :inherited-members:
