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
        assert supply.get_voltage() == [u_nominal]

    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        supply = vs.IdealVoltageSupply(u_nominal)
        assert supply.reset() == [u_nominal]
        
class TestRCVoltageSupply(TestVoltageSupply):
    
    key = 'RCVoltageSupply'
    class_to_test = vs.RCVoltageSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.RCVoltageSupply()
        assert voltage_supply._u_0 == 600.0
        assert voltage_supply._u_sup == [600.0]
        assert voltage_supply.supply_range == (0.0, 600.0)
        assert voltage_supply._r == 1
        assert voltage_supply._c == 4e-3
        
    def test_reset(self, u_nominal=450.0):
        """Test the reset function.
            It must return u_nominal."""
        voltage_supply = vs.RCVoltageSupply(u_nominal)
        assert voltage_supply.reset() == [u_nominal]

    def test_initialization(self, u_nominal=450.0,supply_parameter={'R':3,'C':6e-2}):
        voltage_supply = vs.RCVoltageSupply(u_nominal, supply_parameter)
        assert voltage_supply._u_0 == u_nominal
        assert voltage_supply._u_sup == [u_nominal]
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
            assert supply.get_voltage(time,0) == [u_nominal + time]
            assert supply._u_sup == u_nominal + time
            
    def test_system_equation(self):
         """Tests the correct behavior of the system equation by hand calculated values"""
         supply = vs.RCVoltageSupply()
         system_equation = supply.system_equation
         assert system_equation(0,[10],50,1,1,1) == 39
         assert system_equation(0,[20],50,2,2,2) == 6.5
         assert system_equation(0,[30],50,3,3,3) == 1 + 2/9
         #time invariance
         assert system_equation(0,[30],50,3,3,3) == system_equation(5,[30],50,3,3,3)     


class TestAC1PhaseSupply(TestVoltageSupply):
    key = 'AC1PhaseSupply'
    class_to_test = vs.AC1PhaseSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.AC1PhaseSupply()
        assert voltage_supply.u_nominal == 230.0
        assert voltage_supply._max_amp == 230.0 * np.sqrt(2)
        assert voltage_supply.supply_range == [-230.0 * np.sqrt(2), 230.0 * np.sqrt(2)]
        assert voltage_supply._fixed_phi == False
        assert voltage_supply._f == 50
        
    def test_reset(self, u_nominal=230.0):
        """Test the reset function for correct behavior on fixed phase""" 
        supply_parameter = {'frequency': 50}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        first_phi = voltage_supply._phi
        _ = voltage_supply.reset()
        assert voltage_supply._phi != first_phi

        supply_parameter = {'frequency': 50, 'phase': 0}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply._phi == 0
        assert voltage_supply.reset() == [0.0]
        assert voltage_supply._phi == 0

    def test_initialization(self, u_nominal = 300.0):
        supply_parameter = {'frequency': 35, 'phase': 0}
        voltage_supply = vs.AC1PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply.u_nominal == 300.0
        assert voltage_supply._max_amp == 300.0 * np.sqrt(2)
        assert voltage_supply.supply_range == [-300.0 * np.sqrt(2), 300.0 * np.sqrt(2)]
        assert voltage_supply._fixed_phi == True
        assert voltage_supply._phi == 0
        assert voltage_supply._f == 35
        
    def test_get_voltage(self):
        """Test the get voltage function for different times t."""
        supply_parameter = {'frequency': 1, 'phase': 0}
        supply = vs.AC1PhaseSupply(supply_parameter=supply_parameter)
        
        # Test for default sinus values
        times = [0, 1, 2]
        for time in times:
            assert np.allclose(supply.get_voltage(time), [0.0])
            
        times = [1/4, 5/4, 9/4]
        for time in times:
            assert np.allclose(supply.get_voltage(time),[230.0 * np.sqrt(2)])
            
        times = [3/4, 7/4, 11/4]
        for time in times:
            assert np.allclose(supply.get_voltage(time),[-230.0 * np.sqrt(2)])
          
        # manually calculated
        supply_parameter = {'frequency': 36, 'phase': 0.5}
        supply = vs.AC1PhaseSupply(supply_parameter=supply_parameter)
        assert np.allclose(supply.get_voltage(1/(2*np.pi)),[-303.058731])
        assert np.allclose(supply.get_voltage(2/(2*np.pi)),[-78.381295])
        assert np.allclose(supply.get_voltage(3/(2*np.pi)),[323.118651])


class TestAC3PhaseSupply(TestVoltageSupply):
    key = 'AC3PhaseSupply'
    class_to_test = vs.AC3PhaseSupply
    
    def test_default_initialization(self):
        """Test for default initialization values"""
        voltage_supply = vs.AC3PhaseSupply()
        assert voltage_supply.u_nominal == 400.0
        assert voltage_supply._max_amp == 400.0 / np.sqrt(3) * np.sqrt(2) 
        assert voltage_supply.supply_range == [-400.0 / np.sqrt(3) * np.sqrt(2), 400.0 / np.sqrt(3) * np.sqrt(2)]
        assert voltage_supply._fixed_phi == False
        assert voltage_supply._f == 50
        
    def test_reset(self, u_nominal=400.0):
        """Test the reset function for correct behavior on fixed phase""" 
        supply_parameter = {'frequency': 50}
        voltage_supply = vs.AC3PhaseSupply(u_nominal, supply_parameter)
        first_phi = voltage_supply._phi
        _ = voltage_supply.reset()
        assert voltage_supply._phi != first_phi, "Test this again and if this error doesn't appear next time you should consider playing lotto"

        supply_parameter = {'frequency': 50, 'phase': 0}
        voltage_supply = vs.AC3PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply._phi == 0
        _ = voltage_supply.reset()
        assert voltage_supply._phi == 0

    def test_initialization(self, u_nominal = 300.0):
        supply_parameter = {'frequency': 35, 'phase': 0}
        voltage_supply = vs.AC3PhaseSupply(u_nominal, supply_parameter)
        assert voltage_supply.u_nominal == 300.0
        assert voltage_supply._max_amp == 300.0 / np.sqrt(3) * np.sqrt(2)
        assert voltage_supply.supply_range == [-300.0 / np.sqrt(3) * np.sqrt(2), 300.0 / np.sqrt(3) * np.sqrt(2)]
        assert voltage_supply._fixed_phi == True
        assert voltage_supply._phi == 0
        assert voltage_supply._f == 35
        
    def test_get_voltage(self):
        """Test the get voltage function for different times t."""
        supply_parameter = {'frequency': 1, 'phase': 0}
        supply = vs.AC3PhaseSupply(supply_parameter=supply_parameter)
        
        assert len(supply.get_voltage(0)) == 3
        
        # Test for default sinus values
        times = [0, 1, 2]
        for time in times:
            assert np.allclose(supply.get_voltage(time)[0], [0.0])
            
        times = [1/4, 5/4, 9/4]
        for time in times:
            assert np.allclose(supply.get_voltage(time)[0], 400.0 / np.sqrt(3) * np.sqrt(2))
            
        times = [3/4, 7/4, 11/4]
        for time in times:
            assert np.allclose(supply.get_voltage(time)[0], -400.0 / np.sqrt(3) * np.sqrt(2))
    
        # Manually calculated
        supply_parameter = {'frequency': 41, 'phase': 3.26}
        supply = vs.AC3PhaseSupply(supply_parameter=supply_parameter)
        assert np.allclose(supply.get_voltage(1/(2*np.pi)),[89.536111,227.238311, -316.774422])
        assert np.allclose(supply.get_voltage(2/(2*np.pi)),[-138.223662,-187.151049, 325.374712])

