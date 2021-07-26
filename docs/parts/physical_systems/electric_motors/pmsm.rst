Permanent Magnet Synchronous Motor
##################################

Schematic
*********

.. figure:: ../../../plots/ESBdq.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_\mathrm{sd}}{\mathrm{d} t}&=\frac{u_\mathrm{sd} + p \omega_\mathrm{me} L_\mathrm{q} i_\mathrm{sq} - R_\mathrm{s} i_\mathrm{sd}}{L_\mathrm{d}} \\
    \frac{\mathrm{d} i_\mathrm{sq}}{\mathrm{d} t}&=\frac{u_\mathrm{sq} - p \omega_\mathrm{me} (L_\mathrm{d} i_\mathrm{sd} + \mathit{\Psi}_\mathrm{p}) - R_\mathrm{s} i_\mathrm{sq}}{L_\mathrm{q}} \\
    \frac{\mathrm{d} \varepsilon_\mathrm{el}}{\mathrm{d} t}&= p \omega_\mathrm{me}



Torque Equation
***************

.. math:: T=\frac{3}{2} p (\mathit{\Psi}_\mathrm{p} +(L_\mathrm{d}-L_\mathrm{q})i_\mathrm{sd}) i_\mathrm{sq}

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.PermanentMagnetSynchronousMotor
   :members:
   :inherited-members:
