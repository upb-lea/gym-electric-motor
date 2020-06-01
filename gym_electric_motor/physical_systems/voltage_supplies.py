from gym_electric_motor.physical_systems.solvers import EulerSolver

class VoltageSupply:
    """
    Base class for all VoltageSupplies to be used in a SCMLSystem.

    Parameter:
        supply_range(Tuple(float,float)): Minimal and maximal possible value for the voltage supply.
        u_nominal(float): Nominal supply voltage
    """

    #: Minimum and Maximum values of the Supply Voltage.
    supply_range = ()

    @property
    def u_nominal(self):
        """
        Returns:
             float: Nominal Voltage of the Voltage Supply
        """
        return self._u_nominal

    def __init__(self, u_nominal, **__):
        """
        Args:
            u_nominal(float): Nominal voltage of the Voltage Supply.
        """
        self._u_nominal = u_nominal

    def reset(self):
        """
        Reset the voltage supply to an initial state.
        This method is called at every reset of the physical system.

        Returns:
            float: The initial supply voltage.
        """
        return self.get_voltage(0, 0)

    def get_voltage(self, i_sup, t, *args, **kwargs):
        """
        Get the supply voltage based on the floating supply current i_sup, the time t and optional further arguments.

        Args:
            i_sup(float): Supply current floating into the system.
            t(float): Current time of the system.

        Returns:
             float: Supply Voltage at time t.
        """
        raise NotImplementedError


class IdealVoltageSupply(VoltageSupply):
    """
    Ideal Voltage Supply that supplies with u_nominal independent of the time and the supply current.
    """

    def __init__(self, u_nominal=600.0, **__):
        # Docstring of superclass
        super().__init__(u_nominal)
        self.supply_range = (u_nominal, u_nominal)

    def get_voltage(self, *_, **__):
        # Docstring of superclass
        return self._u_nominal


class RCVoltageSupply(VoltageSupply):
    """Voltage supply moduled as RC element"""
    
    def __init__(self, u_nominal, R, C, **__):
        super().__init__(u_nominal)
        self.supply_range = (0,u_nominal) #???
        #self._R = R
        #self._C = C
        self._solver = EulerSolver()
        self._solver.set_f_params(u_nominal,R,C)
        
        def system_equation(t, u_out, u_nominal, R, C):
            return (C*u_nominal - u_out)/(R*C)
            
        self._solver.set_system_equation(system_equation)
                
    
    def get_voltage(self, i_sup, t):
        
        return self._solver.integrate(t)
