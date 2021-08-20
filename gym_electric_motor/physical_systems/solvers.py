from scipy.integrate import ode, solve_ivp, odeint


class OdeSolver:
    """
    Interface and base class for all used OdeSolvers in gym-electric-motor.
    """

    #: Current system time t
    _t = 0
    #: Current system state y
    _y = None
    #: Function parameters that are passed to the system equation and the system jacobian additionally to t and y
    _f_params = None
    #: System equation in the form: _system_equation(t, y, *f_params)
    _system_equation = None
    #: System jacobian in the form _system_jacobian(t,y, *f_params)
    _system_jacobian = None

    @property
    def t(self):
        """
        Returns:
            float: Current system time t
        """
        return self._t

    @property
    def y(self):
        """
        Returns:
            float: Current system state y
        """
        return self._y

    def set_initial_value(self, initial_value, t=0):
        """
        Set the new initial system state after reset.

        Args:
            initial_value(numpy.ndarray(float)): Initial system state
            t(float):  Initial system time
        """
        self._y = initial_value
        self._t = t

    def integrate(self, t):
        """
        Integrate the ODE-System from current time until time t

        Args:
            t(float): Time until the system shall be integrated
        Returns:
            ndarray(float): New system state at time t
        """
        raise NotImplementedError

    def set_system_equation(self, system_equation, jac=None):
        """
        Setting of the systems equation.

        Args:
            system_equation(function_pointer): Pointer to the systems equation with the parameters (t, y, *args)
            jac(function_pointer): Pointer to the systems jacobian with the parameters (t, y, *args)
        """
        self._system_equation = system_equation
        self._system_jacobian = jac

    def set_f_params(self, *args):
        """
        Set further arguments for the systems function call like input quantities.

        Args:
            args(list): Additional arguments for the next function calls.
        """
        self._f_params = args


class EulerSolver(OdeSolver):
    """
    Solves a system of differential equations of first order for a given time step with linear approximation.

        .. math:
            x^\prime(t) = f(x(t))

        .. math:
            x(t + \\frac{\\tau}{nsteps}) = x(t) + x^\prime(t) * \\frac{\\tau}{nsteps}
    """

    def __init__(self, nsteps=1):
        """
        Args:
            nsteps(int): Number of cycles to calculate for each iteration. Higher steps make the system more accurate,
                but take also longer to compute.
        """
        self._nsteps = nsteps
        self._integrate = self._integrate_one_step if nsteps == 1 else self._integrate_nsteps

    def integrate(self, t):
        # Docstring of superclass
        return self._integrate(t)

    def _integrate_nsteps(self, t):
        """
        Integration method for nsteps > 1

        Args:
            t(float): Time until the system shall be calculated

        Returns:
            ndarray(float):The new state of the system.
        """
        tau = (t - self._t) / self._nsteps
        state = self._y
        current_t = t
        for _ in range(self._nsteps):
            delta = self._system_equation(current_t + tau, state, *self._f_params) * tau
            state = state + delta
            current_t += tau
        self._y = state
        self._t = t
        return self._y

    def _integrate_one_step(self, t):
        """
        Integration method for nsteps = 1. (For faster computation)

        Args:
            t(float): Time until the system shall be calculated

        Returns:
            ndarray(float):The new state of the system.
        """
        self._y = self._y + self._system_equation(self._t, self._y, *self._f_params) * (t - self._t)
        self._t = t
        return self._y


class ScipyOdeSolver(OdeSolver):
    """
    Wrapper class for all ode-solvers in the scipy.integrate.ode package.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
    """

    #: Integrator object
    _ode = None

    @property
    def t(self):
        return self._ode.t

    @property
    def y(self):
        return self._ode.y

    def __init__(self, integrator='dopri5', **kwargs):
        """
        Args:
            integrator(str): String to choose the integrator from the scipy.integrate.ode
            kwargs(dict): All parameters that can be set in the "set_integrator"-method of scipy.integrate.ode
        """
        self._solver = None
        self._solver_args = kwargs
        self._integrator = integrator

    def set_system_equation(self, system_equation, jac=None):
        # Docstring of superclass
        super().set_system_equation(system_equation, jac)
        self._ode = ode(system_equation, jac).set_integrator(self._integrator, **self._solver_args)

    def set_initial_value(self, initial_value, t=0):
        # Docstring of superclass
        self._ode.set_initial_value(initial_value, t)

    def set_f_params(self, *args):
        # Docstring of superclass
        super().set_f_params(*args)
        self._ode.set_f_params(*args)
        self._ode.set_jac_params(*args)

    def integrate(self, t):
        # Docstring of superclass
        return self._ode.integrate(t)


class ScipySolveIvpSolver(OdeSolver):
    """
    Wrapper class for all ode-solvers in the scipy.integrate.solve_ivp function

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """

    def __init__(self, **kwargs):
        # Docstring of superclass
        self._solver_kwargs = kwargs

    def set_system_equation(self, system_equation, jac=None):
        # Docstring of superclass
        method = self._solver_kwargs.get('method', None)
        super().set_system_equation(system_equation, jac)

        # Only Radau BDF and LSODA support the jacobian.
        if method in ['Radau', 'BDF', 'LSODA']:
            self._solver_kwargs['jac'] = self._system_jacobian

    def integrate(self, t):
        # Docstring of superclass
        result = solve_ivp(
            self._system_equation, [self._t, t], self._y, t_eval=[t], args=self._f_params, **self._solver_kwargs
        )
        self._t = t
        self._y = result.y.T[-1]
        return self._y


class ScipyOdeIntSolver(OdeSolver):
    """
    Wrapper class for all ode-solvers in the scipy.integrate.odeint function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
    """

    def __init__(self, **kwargs):
        """
        Args:
            kwargs(dict): Arguments to pass to the solver. See the scipy description for further information.
        """
        self._solver_args = kwargs

    def integrate(self, t):
        # Docstring of superclass
        result = odeint(self._system_equation, self._y, [self._t, t], args=self._f_params, Dfun=self._system_jacobian,
                        tfirst=True, **self._solver_args)
        self._t = t
        self._y = result[-1]
        return self._y
