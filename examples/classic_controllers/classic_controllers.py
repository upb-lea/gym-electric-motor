from gym.spaces import Discrete, Box, MultiDiscrete
import sys
import os
from gym_electric_motor.physical_systems import SynchronousMotorSystem, DcMotorSystem
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))


class Controller:

    @classmethod
    def make(cls, controller_type, environment, **controller_kwargs):
        assert controller_type in _controllers.keys(), f'Controller {controller_type} unknown'
        controller = _controllers[controller_type][0](controller_type, environment, **controller_kwargs)
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass
        
class ContinuousActionController(Controller):
    def __init__(self, controller_type, environment, **kwargs):
        self.action_space = environment.action_space
        assert type(self.action_space) is Box and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Continuous Action Controller'
        self.ref_state_idx = np.where(environment.reference_generator.referenced_states == True)[0]
        self.controller = _controllers[controller_type][1].make(controller_type, environment, **kwargs)
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        if 'psi_e' in self.mp.keys():
            self.psi_e = self.mp['psi_e']
        else:
            self.l_e = self.mp['l_e_prime']
            self.psi_e = None
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]
        self.omega_idx = environment.state_names.index('omega')
        self.action = np.zeros(self.action_space.shape[0])
        if self.action_space.shape[0] > 1:
            self.action[1] = self.limit[self.i_idx] * self.mp['r_e'] / self.limit[self.u_idx]

    def control(self, state, reference):
        self.action[0] = self.controller.control(state[self.ref_state_idx], reference[0]) + self.decoupling(state)
        if self.action_space.low[0] <= self.action.all() <= self.action_space.high[0]:
            self.controller.integrate(state[self.ref_state_idx], reference[0])
        return np.clip(self.action, self.action_space.low, self.action_space.high)

    def reset(self):
        self.controller.reset()

    def decoupling(self, state):
        psi_e = self.psi_e or self.l_e * state[self.i_idx] * self.limit[self.i_idx]
        return (state[self.omega_idx] * self.limit[self.omega_idx] * psi_e) / self.limit[self.u_idx]

class DiscreteActionController(Controller):
    def __init__(self, controller_type, environment, **kwargs):
        assert type(environment.action_space) in [Discrete, MultiDiscrete] and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Discrete Action Controller'
        self.action_space = environment.action_space
        self.limit = environment.physical_system.limits
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        if self.action_space.shape != ():
            self.action = np.zeros(self.action_space.nvec.shape, dtype='int')
            self.k = 0
            self.u_e_limit = int(self.limit[self.u_idx] / (self.limit[self.i_idx] * self.mp['r_e'])) + 1
        self.controller = _controllers[controller_type][1].make(controller_type, environment, **kwargs)
        self.ref_state_idx = np.where(environment.reference_generator.referenced_states == True)[0][0]

    def control(self, state, reference):
        if self.action_space.shape == ():
            return self.controller.control(state[self.ref_state_idx], reference[0])
        else:
            action_u_e = 1 if self.k % self.u_e_limit == 0 else 0
            self.k += 1
            return [self.controller.control(state[self.ref_state_idx], reference[0]), action_u_e]

    def reset(self):
        self.controller.reset()
        self.k = 0

class Cascaded_Controller(Controller):
    def __init__(self, controller_type, environment, outer_controller, inner_controller, **kwargs):
        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space
        self.ref_outer_state_idx = np.where(environment.reference_generator.referenced_states == True)[0][0]
        assert environment.state_names[self.ref_outer_state_idx] == 'omega' and type(
            environment.physical_system) is DcMotorSystem, 'No suitable reference state for Cascaded Controller'

        self.ref_inner_state_idx = environment.physical_system.CURRENTS_IDX[0]
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]

        if 'psi_e' in self.mp.keys():
            self.psi_e = self.mp['psi_e']
        else:
            self.l_e = self.mp['l_e_prime']
            self.psi_e = None

        self.outer_type = _controllers[outer_controller['controller_type']][1] == ContinuousController
        self.outer_controller = _controllers[outer_controller['controller_type']][1].make(
            outer_controller['controller_type'], environment, param_dict=outer_controller, cascaded='outer')

        self.inner_type = _controllers[inner_controller['controller_type']][1] == ContinuousController

        if self.inner_type:
            assert type(self.action_space) is Box, 'No suitable inner controller'
        else:
            assert type(self.action_space) in [Discrete, MultiDiscrete], 'No suitable inner controller'

        self.inner_controller = _controllers[inner_controller['controller_type']][1].make(
            inner_controller['controller_type'], environment, param_dict=inner_controller, cascaded='inner')
        if type(self.action_space) == MultiDiscrete:
            self.k = 0
            self.u_e_limit = int(self.limit[self.u_idx] / (self.limit[self.i_idx] * self.mp['r_e'])) + 1

    def control(self, state, reference):

        psi_e = max(self.psi_e or self.l_e * state[self.i_idx] * self.limit[self.i_idx], 1e-6)

        self.ref_i = self.outer_controller.control(state[self.ref_outer_state_idx], reference[0])
        if (0.85 * self.state_space.low[self.ref_inner_state_idx] <= self.ref_i <= 0.85 * self.state_space.high[
            self.ref_inner_state_idx]) and self.outer_type:
            self.outer_controller.integrate(state[self.ref_outer_state_idx], reference[0])
        elif self.outer_type:
            self.ref_i = np.clip(self.ref_i, 0.85 * self.state_space.low[self.ref_inner_state_idx],
                                 0.85 * self.state_space.high[self.ref_inner_state_idx])

        decoupling = self.decoupling(state, psi_e) if self.inner_type else 0
        action = self.inner_controller.control(state[self.ref_inner_state_idx], self.ref_i) + decoupling

        if self.inner_type and (self.action_space.low[0] <= action.all() <= self.action_space.high[0]):
            self.inner_controller.integrate(state[self.ref_inner_state_idx], self.ref_i)
        elif self.inner_type:
            action = np.clip(action, self.action_space.low[0], self.action_space.high[0])

        if self.action_space.shape == (2,):
            if self.inner_type:
                action = np.array([action, self.limit[self.i_idx] * self.mp['r_e'] / self.limit[self.u_idx]],
                                  dtype='object')
            else:
                action_u_e = 1 if self.k % self.u_e_limit == 0 else 0
                self.k += 1
                action = np.array([action, action_u_e], dtype='object')
        return action
     
    def decoupling(self, state, psi_e):
        return (state[self.ref_outer_state_idx] * self.limit[self.ref_outer_state_idx] * psi_e) / self.limit[self.u_idx]

    def reset(self):
        self.outer_controller.reset()
        self.inner_controller.reset()
        self.k = 0
        
class ContinuousController:
    @classmethod
    def make(cls, controller_type, environment, **controller_kwargs):
        controller = _controllers[controller_type][2](environment, controller_type, **controller_kwargs)
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass


class P_Controller(ContinuousController):
    def __init__(self, p_gain=5):
        self.p_gain = p_gain


class I_Controller(ContinuousController):
    def __init__(self, i_gain=10):
        self.i_gain = i_gain
        self.integrated = 0

    def integrate(self, state, reference):
        self.integrated += (reference - state) * self.tau


class D_Controller(ContinuousController):
    def __init__(self, d_gain=0.05):
        self.d_gain = d_gain
        self.e_old = 0


class PI_Controller(P_Controller, I_Controller):
    def __init__(self, environment, controller_type, p_gain=5, i_gain=5, param_dict=dict(), cascaded=None):
        self.tau = environment.physical_system.tau
        if 'p_gain' in param_dict.keys(): p_gain = param_dict['p_gain']
        if 'i_gain' in param_dict.keys(): i_gain = param_dict['i_gain']
        P_Controller.__init__(self, p_gain)
        I_Controller.__init__(self, i_gain)

    def control(self, state, reference):

        return [self.p_gain * (reference - state) + self.i_gain * (self.integrated + (reference - state) * self.tau)]

    def reset(self):
        self.integrated = 0


class PID_Controller(PI_Controller, D_Controller):
    def __init__(self, environment, controller_type, p_gain=5, i_gain=5, d_gain=0.005, param_dict=dict(),
                 cascaded=None):
        if 'p_gain' in param_dict.keys(): p_gain = param_dict['p_gain']
        if 'i_gain' in param_dict.keys(): i_gain = param_dict['i_gain']
        if 'd_gain' in param_dict.keys(): d_gain = param_dict['d_gain']

        PI_Controller.__init__(self, environment, controller_type, p_gain, i_gain)
        D_Controller.__init__(self, d_gain)

    def control(self, state, reference):
        action = PI_Controller.control(self, state, reference) + self.d_gain * (
                    reference - state - self.e_old) / self.tau
        self.e_old = reference - state
        return action

    def reset(self):
        PI_Controller.reset(self)
        self.e_old = 0


class DiscreteController:
    @classmethod
    def make(cls, controller_type, environment, **controller_kwargs):
        if type(environment.action_space) == Discrete:
            action_space_n = environment.action_space.n
        elif type(environment.action_space) == MultiDiscrete:
            action_space_n = environment.action_space.nvec[0]
        else:
            action_space_n = 3

        controller = _controllers[controller_type][2](environment, action_space=action_space_n, **controller_kwargs)
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass


class OnOff_Controller(DiscreteController):
    def __init__(self, environment, action_space, hysteresis=0.01, param_dict=dict(), cascaded=None):
        self.hysteresis = hysteresis if 'hysteresis' not in param_dict.keys() else param_dict['hysteresis']
        self.switch_on_level = 1
        self.switch_off_level = 2 if action_space in [3, 4] else 0
        if cascaded == 'outer':
            self.switch_off_level = int(environment.physical_system.state_space.low[0])

        self.action = self.switch_on_level

    def control(self, state, reference):
        if reference - state > self.hysteresis:
            self.action = self.switch_on_level

        elif reference - state < self.hysteresis:
            self.action = self.switch_off_level

        return self.action

    def reset(self):
        self.action = self.switch_on_level


class ThreePoint_Controller(DiscreteController):
    def __init__(self, environment, action_space, switch_to_positive_level=0.01, switch_to_negative_level=0.01,
                 switch_to_neutral_from_positive=0.005, switch_to_neutral_from_negative=0.005, param_dict=dict(),
                 cascaded=None):

        self.pos = switch_to_positive_level if 'switch_to_positive_level' not in param_dict.keys() else param_dict[
            'switch_to_positive_level']
        self.neg = switch_to_negative_level if 'switch_to_negative_level' not in param_dict.keys() else param_dict[
            'switch_to_negative_level']
        self.neutral_from_pos = switch_to_neutral_from_positive if 'switch_to_neutral_from_positive' not in param_dict.keys() else \
            param_dict['switch_to_neutral_from_positive']
        self.neutral_from_neg = switch_to_neutral_from_negative if 'switch_to_neutral_from_negative' not in param_dict.keys() else \
            param_dict['switch_to_neutral_from_negative']

        self.negative = 2 if action_space in [3, 4] else 0
        if cascaded == 'outer':
            self.negative = int(environment.physical_system.state_space.low[0])
        self.positive = 1
        self.neutral = 0

        self.action = self.neutral
        self.recent_action = self.neutral

    def control(self, state, reference):
        if reference - state > self.pos or ((self.neutral_from_pos < reference - state) and self.recent_action == 1):
            self.action = self.positive
            self.recent_action = 1
        elif reference - state < -self.neg or (
                (-self.neutral_from_neg > reference - state) and self.recent_action == 2):
            self.action = self.negative
            self.recent_action = 2
        else:
            self.action = self.neutral
            self.recent_action = 0

        return self.action

    def reset(self):
        self.action = self.neutral
        self.recent_action = self.neutral


_controllers = {
    'pi_controller': [ContinuousActionController, ContinuousController, PI_Controller],
    'pid_controller': [ContinuousActionController, ContinuousController, PID_Controller],
    'on_off': [DiscreteActionController, DiscreteController, OnOff_Controller],
    'three_point': [DiscreteActionController, DiscreteController, ThreePoint_Controller],
    'cascaded_controller': [Cascaded_Controller],
}
