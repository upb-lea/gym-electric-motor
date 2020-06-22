import gym_electric_motor as gem
import gym_electric_motor.physical_systems.voltage_supplies as vs
from ..testing_utils import DummyOdeSolver
import numpy as np

class TestVoltageSupply:

    key = ''
    class_to_test = vs.VoltageSupply

    def test_registered(self):
        """If a key is provided, tests if the class can be initialized from registry"""
        if self.key != '':
            supply = gem.utils.instantiate(vs.VoltageSupply, self.key)
            assert type(supply) == self.class_to_test

    def test_initialization(self):
        """Test the initalization and correct setting of values"""
        u_nominal = 600.0
        voltage_supply = self.class_to_test(u_nominal)
        assert voltage_supply.u_nominal == u_nominal


class TestIdealVoltageSupply(TestVoltageSupply):

    key = 'IdealVoltageSupply'
    class_to_test = vs.IdealVoltageSupply

    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.IdealVoltageSupply()
        assert voltage_supply._u_nominal == 600.0
        assert voltage_supply.supply_range == (600.0, 600.0)

    def test_get_voltage(self, u_nominal=450.0):
        """Test the get voltage function.
            It must return u_nominal."""
        supply = vs.IdealVoltageSupply(u_nominal)
        assert supply.get_voltage() == u_nominal

    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        supply = vs.IdealVoltageSupply(u_nominal)
        assert supply.reset() == u_nominal
        
class TestRCVoltageSupply(TestVoltageSupply):
    
    key = 'RCVoltageSupply'
    class_to_test = vs.RCVoltageSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.RCVoltageSupply()
        assert voltage_supply._u_0 == 600.0
        assert voltage_supply._u_sup == 600.0
        assert voltage_supply.supply_range == (0.0, 600.0)
        assert voltage_supply._r == 1
        assert voltage_supply._c == 4e-3
        
    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        voltage_supply = vs.RCVoltageSupply(u_nominal)
        assert voltage_supply.reset() == u_nominal

        
    def test_initialization(self, u_nominal=450.0,supply_parameter={'R':3,'C':6e-2}):
        voltage_supply = vs.RCVoltageSupply(u_nominal, supply_parameter)
        assert voltage_supply._u_0 == u_nominal
        assert voltage_supply._u_sup == u_nominal
        assert voltage_supply.supply_range == (0.0, u_nominal) 
        assert voltage_supply._r == supply_parameter['R']
        assert voltage_supply._c == supply_parameter['C']
        
    def test_get_voltage(self, monkeypatch, u_nominal=450.0):
        """Test the get voltage function.
            It must return the right voltage added by its change given by the time delta in this example."""
        supply = vs.RCVoltageSupply(u_nominal)
        solver = DummyOdeSolver()
        solver._y = np.array([u_nominal])
        monkeypatch.setattr(supply, '_solver', solver)
        times = [0.5,1,1.5,1.78,2.1,3]
        for time in times:
            assert supply.get_voltage(time,0) == u_nominal + time
            assert supply._u_sup == u_nominal + time
            
    def test_system_equation(self):
        """Tests the correct behavior of the system equation in the get_voltage() function"""
        def system_equation(t, u_sup, u_0, i_sup, r, c):
            # ODE for derivate of u_sup
            return np.array([(u_0 - u_sup[0] - r*i_sup)/(r*c)])
        supply = vs.RCVoltageSupply()
        i_sups = [0,2,4,6,8,10]
        ts = [1e-3,1e-2,1,10,100]
        for i_sup,t in zip(i_sups,ts):
            supply.reset()
            expected_diff = system_equation(t,[supply._u_sup], supply._u_0, i_sup, supply._r, supply._c) * t
            voltage = supply.get_voltage(t,i_sup)
            assert voltage - supply._u_0 == expected_diff




    
    
