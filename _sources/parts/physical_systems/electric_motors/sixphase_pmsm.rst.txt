Six Phase Permanent Magnet Synchronous Motor
##################################


Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_s^d}{\mathrm{d} t} = \frac{u_s^d + \omega_\mathrm{el} L_s^q i_s^q - R_s i_s^d}{L_s^d} \\
    \frac{\mathrm{d} i_s^q}{\mathrm{d} t} = \frac{u_s^q - \omega_\mathrm{el} L_s^d i_s^d - R_s i_s^q - \omega_\mathrm{el}}{L_s^q} \\
    \frac{\mathrm{d} i_s^x}{\mathrm{d} t} = \frac{u_s^x - \omega_\mathrm{el} L_s^y i_s^y - R_s i_s^x}{L_s^x} \\
    \frac{\mathrm{d} i_s^y}{\mathrm{d} t} = \frac{u_s^y + \omega_\mathrm{el} L_s^x i_s^x - R_s i_s^y}{L_s^y} \\



Torque Equation
***************

.. math:: T=\frac{3}{2} p (\mathit{\Psi}_\mathrm{p} +(L_\mathrm{d}-L_\mathrm{q})i_\mathrm{sd}) i_\mathrm{sq}

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SixPhasePMSM
   :members:
   :inherited-members:
