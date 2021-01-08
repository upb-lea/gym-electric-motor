from gym.spaces import Discrete, Box, MultiDiscrete
from gym_electric_motor.physical_systems import SynchronousMotorSystem, DcMotorSystem
import numpy as np


class Controller:

    @classmethod
    def make(cls, environment, controller_type=None, **controller_kwargs):

        ref_states = []
        for i in range(np.size(np.where(environment.reference_generator.referenced_states == True)[0])):
            ref_states.append(
                environment.state_names[np.where(environment.reference_generator.referenced_states == True)[0][i]])

        if controller_type != None:

            assert controller_type in _controllers.keys(), f'Controller {controller_type} unknown'
            controller_kwargs = cls.automated_gain(cls, controller_type, environment, ref_states, **controller_kwargs)
            controller = _controllers[controller_type][0](controller_type, environment, **controller_kwargs)

        else:

            controller_type, controller_kwargs = cls.automated_controller_design(cls, environment, controller_type,
                                                                                 ref_states, **controller_kwargs)
            controller_kwargs = cls.automated_gain(cls, controller_type, environment, ref_states, **controller_kwargs)
            controller = _controllers[controller_type][0](controller_type, environment, **controller_kwargs)

        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass

    def automated_controller_design(self, environment, controller_type, ref_states, **controller_kwargs):
        action_space = type(environment.action_space)

        if type(environment.physical_system) == DcMotorSystem:

            if 'omega' in ref_states:
                controller_type = 'cascaded_controller'
                stages = controller_kwargs['stages'] if 'stages' in controller_kwargs.keys() else []
                for i in range(len(stages), 2):
                    if i == 0:
                        if action_space == Box:
                            stages.append({'controller_type': 'pi_controller'})
                        else:
                            stages.append({'controller_type': 'three_point'})
                    else:
                        stages.append({'controller_type': 'pi_controller'})

                controller_kwargs['stages'] = stages

            elif 'i' in ref_states or 'i_a' in ref_states or 'torque' in ref_states:

                if action_space == Discrete or action_space == MultiDiscrete:
                    controller_type = 'three_point'
                elif action_space == Box:
                    controller_type = 'pi_controller'


        elif type(environment.physical_system) == SynchronousMotorSystem:
            controller_type = 'foc_controller'
            if 'q_controller' not in controller_kwargs.keys():
                controller_kwargs['q_controller'] = {'controller_type': 'pi_controller'}
            if 'd_controller' not in controller_kwargs.keys():
                controller_kwargs['d_controller'] = {'controller_type': 'pi_controller'}

        return controller_type, controller_kwargs

    def automated_gain(self, controller_type, environment, ref_states, **controller_kwargs):
        mp = environment.physical_system.electrical_motor.motor_parameter
        cv_idx = environment.state_names.index('u') if 'u' in environment.state_names else environment.state_names.index('u_a')
        i_idx = environment.state_names.index(
            'i') if 'i' in environment.state_names else environment.state_names.index('i_a')
        a = 2 if 'a' not in controller_kwargs.keys() else controller_kwargs['a']
        automated_gain = True if 'automated_gain' not in controller_kwargs.keys() else controller_kwargs[
            'automated_gain']

        if 'automated_gain' not in controller_kwargs.keys() or automated_gain:
            if _controllers[controller_type][0] == ContinuousActionController:
                if 'i' in ref_states or 'i_a' in ref_states or 'torque' in ref_states:
                    p_gain = mp['l_a'] / (environment.physical_system.tau * a) / environment.physical_system.limits[cv_idx]
                    i_gain = p_gain / (environment.physical_system.tau * a**2)

                    controller_kwargs['p_gain'] = p_gain if 'p_gain' not in controller_kwargs.keys() else \
                        controller_kwargs['p_gain']

                    controller_kwargs['i_gain'] = i_gain if 'i_gain' not in controller_kwargs.keys() else \
                        controller_kwargs['i_gain']

                    if _controllers[controller_type][2] == PID_Controller:
                        d_gain = p_gain * environment.physical_system.tau
                        controller_kwargs['d_gain'] = d_gain if 'd_gain' not in controller_kwargs.keys() else \
                            controller_kwargs['d_gain']

                elif 'omega' in ref_states:
                    p_gain = 32 * environment.physical_system.tau * mp['r_a'] / (a * mp['l_a'] ** 3 * environment.physical_system.limits[cv_idx])
                    i_gain = p_gain / (a ** 2 * mp['l_a'] / mp['r_a'])

                    controller_kwargs['p_gain'] = p_gain if 'p_gain' not in controller_kwargs.keys() else \
                        controller_kwargs['p_gain']
                    controller_kwargs['i_gain'] = i_gain if 'i_gain' not in controller_kwargs.keys() else \
                        controller_kwargs['i_gain']

                    if _controllers[controller_type][2] == PID_Controller:
                        d_gain = p_gain * environment.physical_system.tau
                        controller_kwargs['d_gain'] = d_gain if 'd_gain' not in controller_kwargs.keys() else \
                            controller_kwargs['d_gain']


            elif _controllers[controller_type][0] == Cascaded_Controller:
                stages = controller_kwargs['stages']
                for i in range(len(stages)):

                    if _controllers[stages[i]['controller_type']][1] == ContinuousController:

                        if i == 0:
                            p_gain = mp['l_a'] / (environment.physical_system.tau * a) / \
                                           environment.physical_system.limits[cv_idx]
                            i_gain = p_gain / (environment.physical_system.tau * a ** 2)


                            if _controllers[stages[i]['controller_type']][
                                2] == PID_Controller:
                                d_gain = p_gain * environment.physical_system.tau
                                stages[i]['d_gain'] = d_gain if 'd_gain' not in stages[i].keys() else stages[i][
                                    'd_gain']

                        elif i == 1:
                            p_gain = mp['l_a'] / (mp['r_a'] * environment.physical_system.tau) / environment.physical_system.limits[i_idx]
                            i_gain = p_gain / (a ** 2 * 32 * environment.physical_system.tau)

                            if _controllers[stages[i]['controller_type']][
                                2] == PID_Controller:
                                d_gain = p_gain * environment.physical_system.tau
                                stages[i]['d_gain'] = d_gain if 'd_gain' not in stages[i].keys() else stages[i][
                                    'd_gain']
                    stages[i]['p_gain'] = p_gain if 'p_gain' not in stages[i].keys() else stages[i]['p_gain']
                    stages[i]['i_gain'] = i_gain if 'i_gain' not in stages[i].keys() else stages[i]['i_gain']
                controller_kwargs['stages'] = stages

            elif _controllers[controller_type][0] == FOC_Controller:
                print('Berechne FOC Gain')
        print(controller_kwargs)
        return controller_kwargs




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
        self.action[0] = self.controller.control(state[self.ref_state_idx], reference[0]) + self.feedforward(state)
        if self.action_space.low[0] <= self.action.all() <= self.action_space.high[0]:
            self.controller.integrate(state[self.ref_state_idx], reference[0])
        return np.clip(self.action, self.action_space.low, self.action_space.high)

    def reset(self):
        self.controller.reset()

    def feedforward(self, state):
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
    def __init__(self, controller_type, environment, **controller_kwargs):
        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space
        self.ref_outer_state_idx = np.where(environment.reference_generator.referenced_states == True)[0][0]
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.i_e_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.i_a_idx = environment.physical_system.CURRENTS_IDX[0]
        self.omega_idx = environment.state_names.index('omega')
        self.phi_idx = None
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]
        self.stages = len(controller_kwargs['stages'])
        self.ref_state_idx = [self.i_a_idx, self.omega_idx, self.phi_idx]

        if 'psi_e' in self.mp.keys():
            self.psi_e = self.mp['psi_e']
        else:
            self.l_e = self.mp['l_e_prime']
            self.psi_e = None

        self.controller_stages = []
        self.stage_type = []

        for i in range(self.stages):
            kwargs = controller_kwargs['stages'][i]
            self.stage_type.append(_controllers[kwargs['controller_type']][1] == ContinuousController)
            cascaded = 'outer' if i != 0 else 'inner'
            self.controller_stages.append(
                _controllers[kwargs['controller_type']][1].make(kwargs['controller_type'], environment,
                                                                param_dict=kwargs, cascaded=cascaded))

        if self.stage_type[0]:
            assert type(self.action_space) is Box, 'No suitable inner controller'
        else:
            assert type(self.action_space) in [Discrete, MultiDiscrete], 'No suitable inner controller'


        if type(self.action_space) == MultiDiscrete:
            self.k = 0
            self.u_e_limit = int(self.limit[self.u_idx] / (self.limit[self.i_e_idx] * self.mp['r_e'])) + 1

    def control(self, state, reference):

        psi_e = max(self.psi_e or self.l_e * state[self.i_e_idx] * self.limit[self.i_e_idx], 1e-6)

        ref = np.zeros(self.stages)
        ref[-1] = reference[0]

        for i in range(self.stages-1, 0, -1):
            ref[i-1] = self.controller_stages[i].control(state[self.ref_state_idx[i]], ref[i])
            if (0.85 * self.state_space.low[self.ref_state_idx[i-1]] <= ref[i-1] <= 0.85 * self.state_space.high[
                self.ref_state_idx[i-1]]) and self.stage_type[i]:
                self.controller_stages[i].integrate(state[self.ref_state_idx[i]], reference[0])
            elif self.stage_type[i]:
                ref[i - 1] = np.clip(ref[i - 1], 0.85 * self.state_space.low[self.ref_state_idx[i - 1]],
                                     0.85 * self.state_space.high[self.ref_state_idx[i - 1]])

        action = self.controller_stages[0].control(state[self.ref_state_idx[0]], ref[0])
        if self.stage_type[0]:
            action += self.feedforward(state, psi_e)

            if (self.action_space.low[0] <= action.all() <= self.action_space.high[0]):
                self.controller_stages[0].integrate(state[self.ref_state_idx[0]], ref[0])
                action = [action]
            else:
                action = np.clip([action], self.action_space.low[0], self.action_space.high[0])

        if self.action_space.shape == (2,):
            if self.stage_type[0]:
                return np.array([action[0], self.limit[self.i_e_idx] * self.mp['r_e'] / self.limit[self.u_idx]],
                                  dtype='object')
            else:
                action_u_e = 1 if self.k % self.u_e_limit == 0 else 0
                self.k += 1
                action = np.array([action, action_u_e], dtype='object')

        return action

    def feedforward(self, state, psi_e):
        return (state[self.ref_outer_state_idx] * self.limit[self.ref_outer_state_idx] * psi_e) / self.limit[self.u_idx]

    def reset(self):
        for controller in self.controller_stages:
            controller.reset()
        self.k = 0


class FOC_Controller(Controller):
    def __init__(self, controller_type, environment, d_controller, q_controller, **kwargs):
        assert type(environment.physical_system) is SynchronousMotorSystem, 'No suitable Environment for FOC Controller'

        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self.backward_transformation = (lambda quantities, eps: t32(q(quantities[::-1], eps)))

        self.tau = environment.physical_system.tau
        self.references = np.where(environment.reference_generator.referenced_states == True)[0]
        self.d_idx = self.references[0]
        self.q_idx = self.references[1]
        self.ref_d_idx = np.where(self.references == self.d_idx)[0][0]
        self.ref_q_idx = np.where(self.references == self.q_idx)[0][0]

        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space

        self.i_sd_idx = environment.state_names.index('i_sd')
        self.i_sq_idx = environment.state_names.index('i_sq')
        self.u_sd_idx = environment.state_names.index('u_sd')
        self.u_sq_idx = environment.state_names.index('u_sq')
        self.u_a_idx = environment.state_names.index('u_a')
        self.u_b_idx = environment.state_names.index('u_b')
        self.u_c_idx = environment.state_names.index('u_c')
        self.omega_idx = environment.state_names.index('omega')
        self.eps_idx = environment.state_names.index('epsilon')

        self.limit = environment.physical_system.limits
        self.mp = environment.physical_system.electrical_motor.motor_parameter

        self.d_controller = _controllers[d_controller['controller_type']][1].make(
            d_controller['controller_type'], environment, param_dict=d_controller)
        self.q_controller = _controllers[q_controller['controller_type']][1].make(
            q_controller['controller_type'], environment, param_dict=q_controller)

    def control(self, state, reference):

        u_sd_0 = -state[self.omega_idx] * self.mp['p'] * self.mp['l_q'] * state[self.i_sq_idx] * self.limit[
            self.i_sq_idx] / self.limit[self.u_sd_idx] * self.limit[self.omega_idx]
        u_sq_0 = state[self.omega_idx] * self.mp['p'] * (
                    state[self.i_sd_idx] * self.mp['l_d'] * self.limit[self.u_sd_idx] + self.mp['psi_p']) / self.limit[
                     self.u_sq_idx] * self.limit[self.omega_idx]

        u_sd = self.d_controller.control(state[self.d_idx], reference[self.ref_d_idx]) + u_sd_0
        u_sq = self.q_controller.control(state[self.q_idx], reference[self.ref_q_idx]) + u_sq_0

        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + 1.5 * self.tau * state[self.omega_idx] * \
                    self.limit[self.omega_idx] * self.mp['p']
        action_temp = self.backward_transformation((u_sq, u_sd), epsilon_d)[:, 0]
        action_temp = action_temp - 0.5 * (max(action_temp) + min(action_temp))

        action = np.clip(action_temp, self.action_space.low[0], self.action_space.high[0])
        if action.all() == action_temp.all():
            self.d_controller.integrate(state[self.d_idx], reference[self.ref_d_idx])
            self.q_controller.integrate(state[self.q_idx], reference[self.ref_q_idx])

        return action

    def reset(self):
        None

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
    def __init__(self, environment, controller_type, p_gain=5, i_gain=5, cascaded=None, param_dict=dict(), **controller_kwargs):
        self.tau = environment.physical_system.tau

        p_gain = param_dict['p_gain'] if 'p_gain' in param_dict.keys() else p_gain
        i_gain = param_dict['i_gain'] if 'i_gain' in param_dict.keys() else i_gain
        P_Controller.__init__(self, p_gain)
        I_Controller.__init__(self, i_gain)

    def control(self, state, reference):
        return self.p_gain * (reference - state) + self.i_gain * (self.integrated + (reference - state) * self.tau)

    def reset(self):
        self.integrated = 0


class PID_Controller(PI_Controller, D_Controller):
    def __init__(self, environment, controller_type, p_gain=5, i_gain=5, d_gain=0.005, cascaded=None, param_dict=dict(), **controller_kwargs):

        p_gain = param_dict['p_gain'] if 'p_gain' in param_dict.keys() else p_gain
        i_gain = param_dict['i_gain'] if 'i_gain' in param_dict.keys() else i_gain
        d_gain = param_dict['d_gain'] if 'd_gain' in param_dict.keys() else d_gain

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
    def __init__(self, environment, action_space, hysteresis=0.01, param_dict=dict(), cascaded=None, **controller_kwargs):
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
                 cascaded=None, **controller_kwargs):

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
    'foc_controller': [FOC_Controller],
    #    'cascaded_foc_controller': CascadedFOC_Controller,
    #    'foc_rotor_flux_observer': FOC_Rotor_Flux_Observer

}
