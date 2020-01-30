ODE-Solvers
###########

Solving of ODE-Systems in the form

.. math::
    \frac{\mathrm{d} \mathbf{x}}{\mathrm{d} t}&= f(\mathbf{x}, \mathbf{u}, t)\\


..  toctree::
    :maxdepth: 1
    :caption: Available ODE-Solvers:

    euler
    scipy_solve_ivp
    scipy_ode
    scipy_odeint



ODE-Solver Base Class
'''''''''''''''''''''

.. autoclass:: gym_electric_motor.physical_systems.solvers.OdeSolver
    :members:
