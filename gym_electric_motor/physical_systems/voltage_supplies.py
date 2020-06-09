from gym_electric_motor.physical_systems.solvers import EulerSolver
import warnings
import numpy as np

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

    def get_voltage(self, t, i_sup, *args, **kwargs):
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
    
    def __init__(self, u_nominal, supply_parameter={'R':1,'C':4e-3}, **__):
        """
        RC circuit takes additional values for it's electrical elements.
        Args:
            R(float): Reluctance in Ohm
            C(float): Capacitance in Farad
        Additional notes:
            If the product of R and C get too small the numerical stability of the ODE is not given anymore
            since our time difference tau is only in the range of 10e-3. You might want to consider R as a
            time constant used for numerical stability and only model C realistically
        """
        super().__init__(u_nominal)
        # Supply range is between 0 - capacitor completely unloaded - and u_nominal - capacitor is completely loaded
        self.supply_range = (0,u_nominal) 
        self._R = supply_parameter['R']
        self._C = supply_parameter['C']
        if self._R*self._C < 1e-4:
            warnings.warn("The product of R and C might be too small for the correct calculation of the supply voltage. You might want to consider R as a time constant.")

        self._u_sup = u_nominal
        self._u_0 = u_nominal
        self._solver = EulerSolver()
        def system_equation(t, u_sup, u_0, i_sup, R, C):
            # ODE for derivate of u_sup
            return np.array([(u_0 - u_sup[0] - R*i_sup)/(R*C)])
        self._solver.set_system_equation(system_equation)
        
    def reset(self):
        # On reset the capacitor is unloaded again
        self._solver.set_initial_value(np.array([self._u_0]))
        self._u_sup = self._u_0
        return self._u_sup
                
    
    def get_voltage(self, t,i_sup):
        self._solver.set_f_params(self._u_0, i_sup, self._R, self._C)
        self._u_sup = self._solver.integrate(t)[0]
        return self._u_sup
    
