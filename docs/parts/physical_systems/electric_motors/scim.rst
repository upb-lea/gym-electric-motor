Squirrel Cage Induction Motor
##################################

Schematic
*********

.. figure:: ../../../plots/ESB_SCIM_alphabeta.svg

Electrical ODE
**************

.. math::
    \frac{\mathrm{d} i_\mathrm{s \alpha}}{\mathrm{d} t}&= -\frac{1}{\tau_\sigma} i_\mathrm{s \alpha} + \frac{R_\mathrm{r} L_\mathrm{m}}{\sigma L_\mathrm{r}^2 L_\mathrm{s}} \psi_\mathrm{r \alpha} + p \omega_{\text{me}} \frac{L_\mathrm{m}}{\sigma L_\mathrm{r} L_\mathrm{s}} \psi_\mathrm{r \beta} + \frac{1}{\sigma L_\mathrm{s}} u_\mathrm{s \alpha}\\
    \frac{\mathrm{d} i_\mathrm{s \beta}}{\mathrm{d} t}&= -\frac{1}{\tau_\sigma} i_\mathrm{s \beta} - p \omega_{\text{me}} \frac{L_\mathrm{m}}{\sigma L_\mathrm{r} L_\mathrm{s}}  \psi_\mathrm{r \alpha}  + \frac{R_\mathrm{r} L_\mathrm{m}}{\sigma L_\mathrm{r}^2 L_\mathrm{s}} \psi_\mathrm{r \beta} + \frac{1}{\sigma L_\mathrm{s}} u_\mathrm{s \beta}\\
    \frac{\mathrm{d} \psi_\mathrm{r \alpha}}{\mathrm{d} t}&= \frac{L_\mathrm{m}}{\tau_\mathrm{r}} i_\mathrm{s \alpha} - \frac{1}{\tau_\text{r}} \psi_\mathrm{r \alpha} - p \omega_{\text{me}} \psi_\mathrm{r \beta} \\
    \frac{\mathrm{d} \psi_\mathrm{r \beta}}{\mathrm{d} t}&= \frac{L_\mathrm{m}}{\tau_\mathrm{r}} i_\mathrm{s \beta} + p \omega_{\mathrm{me}} \psi_\mathrm{r \alpha} - \frac{1}{\tau_\mathrm{r}} \psi_\mathrm{r \beta} \\
    \frac{\mathrm{d} \varepsilon_\mathrm{el}}{\mathrm{d} t}&= p \omega_\mathrm{me}

with

.. math::
    L_\mathrm{s} &= L_\mathrm{m} + L_\mathrm{\sigma s}, & L_\mathrm{r} &= L_\mathrm{m} + L_\mathrm{\sigma r}\\
    \sigma &= \frac{L_\mathrm{r} L_\mathrm{s} - L_\mathrm{m}^2}{L_\mathrm{r} L_\mathrm{s}}, & \tau_\mathrm{r} &=\frac{L_\mathrm{r}}{R_\mathrm{r}}, & \tau_\sigma &= \frac{\sigma L_\mathrm{s}}{R_\mathrm{s} + R_\mathrm{r} \frac{L_\mathrm{m}^2}{L_\mathrm{r}^2}}


Torque Equation
***************

.. math:: T=\frac{3}{2} p \frac{L_\mathrm{m}}{L_\mathrm{r}} (\psi_\mathrm{r \alpha} i_\mathrm{s \beta} - \psi_\mathrm{r \beta} i_\mathrm{s \alpha})

Code Documentation
******************

.. autoclass:: gym_electric_motor.physical_systems.electric_motors.SquirrelCageInductionMotor
   :members:
   :inherited-members: