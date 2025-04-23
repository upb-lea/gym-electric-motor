Six Phase Permanent Magnet Synchronous Motor
##################################


Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_\mathrm{d}}{\mathrm{d} t}&= \frac{u_\mathrm{d} + \omega_\mathrm{el} L_\mathrm{q} i_\mathrm{q} - R_\mathrm{s} i_\mathrm{d}}{L_\mathrm{d}} \\
    \frac{\mathrm{d} i_\mathrm{q}}{\mathrm{d} t}&= \frac{u_\mathrm{q} - \omega_\mathrm{el} L_\mathrm{d} i_\mathrm{d} - R_\mathrm{s} i_\mathrm{q} - \omega_\mathrm{el} \psi_\mathrm{PM}}{L_\mathrm{q}} \\
    \frac{\mathrm{d} i_\mathrm{x}}{\mathrm{d} t}&= \frac{u_\mathrm{x} - \omega_\mathrm{el} L_\mathrm{y} i_\mathrm{y} - R_\mathrm{s} i_\mathrm{x}}{L_\mathrm{x}} \\
    \frac{\mathrm{d} i_\mathrm{y}}{\mathrm{d} t}&= \frac{u_\mathrm{y} + \omega_\mathrm{el} L_\mathrm{x} i_\mathrm{x} - R_\mathrm{s} i_\mathrm{y}}{L_\mathrm{y}} \\



Torque Equation
***************

.. math:: T=\frac{3}{2} p (\mathit{\Psi}_\mathrm{p} +(L_\mathrm{d}-L_\mathrm{q})i_\mathrm{sd}) i_\mathrm{sq}

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SixPhasePMSM
   :members:
   :inherited-members:
