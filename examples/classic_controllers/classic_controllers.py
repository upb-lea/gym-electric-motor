from gym.spaces import Discrete, Box, MultiDiscrete
from gym_electric_motor.physical_systems import SynchronousMotorSystem, DcMotorSystem, DcSeriesMotor, \
    DcExternallyExcitedMotor
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, SwitchedReferenceGenerator
from gym_electric_motor import envs
import numpy as np


class Controller:

    @classmethod
    def make(cls, environment, stages=None, **controller_kwargs):

        controller_kwargs = cls.reference_states(environment, **controller_kwargs)

        if stages is not None:
            controller_type, stages = cls.find_controller_type(environment, stages)
            assert controller_type in _controllers.keys(), f'Controller {controller_type} unknown'
            stages = cls.automated_gain(environment, stages, controller_type, **controller_kwargs)
            controller = _controllers[controller_type][0](environment, stages, **controller_kwargs)

        else:
            controller_type, stages = cls.automated_controller_design(environment, **controller_kwargs)
            stages = cls.automated_gain(environment, stages, controller_type, **controller_kwargs)

            controller = _controllers[controller_type][0](environment, stages, **controller_kwargs)

        return controller

    def control(self, state, reference):
        pass

    def plot(self):
        pass

    def reset(self):
        pass

    @staticmethod
    def reference_states(environment, **controller_kwargs):
        ref_states = []
        if type(environment.reference_generator) == MultipleReferenceGenerator:

            for rg in environment.reference_generator._sub_generators:
                if type(rg) == SwitchedReferenceGenerator:
                    ref_states.append(rg._sub_generators[0]._reference_state)
                else:
                    ref_states.append(rg._reference_state)

        elif type(environment.reference_generator) == SwitchedReferenceGenerator:
            ref_states.append(environment.reference_generator._sub_generators[0]._reference_state)
        else:
            ref_states.append(environment.reference_generator._reference_state)
        controller_kwargs['ref_states'] = np.array(ref_states)
        return controller_kwargs

    @staticmethod
    def find_controller_type(environment, stages):
        _stages = stages
        if type(environment.physical_system) == DcMotorSystem:
            if type(stages) == list:
                if len(stages) > 1:
                    if type(stages[0]) == list:
                        stages = stages[0]
                    if len(stages) > 1:
                        controller_type = 'cascaded_controller'
                    else:
                        controller_type = stages[0]['controller_type']
                else:
                    controller_type = stages[0]['controller_type']
            else:
                if type(stages) == dict:
                    controller_type = stages['controller_type']
                    _stages = [stages]
                else:
                    controller_type = stages
                    _stages = [{'controller_type': stages}]
        elif type(environment.physical_system) == SynchronousMotorSystem:
            if len(stages) == 2:
                if len(stages[1]) == 1:
                    controller_type = 'foc_controller'
                else:
                    controller_type = 'cascaded_foc_controller'
            else:
                controller_type = 'cascaded_foc_controller'

        return controller_type, _stages

    @staticmethod
    def automated_controller_design(environment, **controller_kwargs):
        action_space = type(environment.action_space)
        ref_states = controller_kwargs['ref_states']
        stages = []
        if type(environment.physical_system) == DcMotorSystem:

            if 'omega' in ref_states or 'torque' in ref_states:
                controller_type = 'cascaded_controller'

                for i in range(len(stages), 2):
                    if i == 0:
                        if action_space == Box:
                            stages.append({'controller_type': 'pi_controller'})
                        else:
                            stages.append({'controller_type': 'three_point'})
                    else:
                        stages.append({'controller_type': 'pi_controller'})

            elif 'i' in ref_states or 'i_a' in ref_states:

                if action_space == Discrete or action_space == MultiDiscrete:
                    stages.append({'controller_type': 'three_point'})
                elif action_space == Box:
                    stages.append({'controller_type': 'pi_controller'})
                controller_type = stages[0]['controller_type']
            if type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor:
                if action_space == Box:
                    stages = [stages, [{'controller_type': 'pi_controller'}]]
                else:
                    stages = [stages, [{'controller_type': 'three_point'}]]

        elif type(environment.physical_system) == SynchronousMotorSystem:
            if 'i_sq' in ref_states:
                controller_type = 'foc_controller'
                if type(environment.action_space) == Discrete:
                    stages = [[{'controller_type': 'on_off'}], [{'controller_type': 'on_off'}],
                              [{'controller_type': 'on_off'}]]
                else:
                    stages = [[{'controller_type': 'pi_controller'}], [{'controller_type': 'pi_controller'}]]
            elif 'omega' in ref_states or 'torque' in ref_states:
                controller_type = 'cascaded_foc_controller'
                if type(environment.action_space) == Discrete:
                    stages = [[{'controller_type': 'on_off'}], [{'controller_type': 'on_off'}],
                              [{'controller_type': 'on_off'}], [{'controller_type': 'pi_controller'}]]
                else:
                    stages = [[{'controller_type': 'pi_controller'}],
                              [{'controller_type': 'pi_controller'}, {'controller_type': 'pi_controller'}]]

        else:
            controller_type = 'foc_controller'

        return controller_type, stages

    @staticmethod
    def automated_gain(environment, stages, controller_type, **controller_kwargs):

        ref_states = controller_kwargs['ref_states']
        mp = environment.physical_system.electrical_motor.motor_parameter

        limits = environment.physical_system.limits
        omega_lim = limits[environment.state_names.index('omega')]
        if type(environment.physical_system) == DcMotorSystem:
            i_a_lim = limits[environment.physical_system.CURRENTS_IDX[0]]
            i_e_lim = limits[environment.physical_system.CURRENTS_IDX[-1]]
            u_a_lim = limits[environment.physical_system.VOLTAGES_IDX[0]]
            u_e_lim = limits[environment.physical_system.VOLTAGES_IDX[-1]]

        elif type(environment.physical_system) == SynchronousMotorSystem:
            i_sd_lim = limits[environment.state_names.index('i_sd')]
            i_sq_lim = limits[environment.state_names.index('i_sq')]
            u_sd_lim = limits[environment.state_names.index('u_sd')]
            u_sq_lim = limits[environment.state_names.index('u_sq')]

        a = 4 if 'a' not in controller_kwargs.keys() else controller_kwargs['a']

        automated_gain = True if 'automated_gain' not in controller_kwargs.keys() else controller_kwargs[
            'automated_gain']
        if type(environment.physical_system.electrical_motor) == DcSeriesMotor:
            mp['l'] = mp['l_a'] + mp['l_e']
        elif type(environment.physical_system) == DcMotorSystem:
            mp['l'] = mp['l_a']

        if 'automated_gain' not in controller_kwargs.keys() or automated_gain:
            cont_extex_envs = (
                envs.ContSpeedControlDcExternallyExcitedMotorEnv,
                envs.ContCurrentControlDcExternallyExcitedMotorEnv,
                envs.ContTorqueControlDcExternallyExcitedMotorEnv
            )
            finite_extex_envs = (
                envs.FiniteTorqueControlDcExternallyExcitedMotorEnv,
                envs.FiniteSpeedControlDcExternallyExcitedMotorEnv,
                envs.FiniteCurrentControlDcExternallyExcitedMotorEnv
            )
            if type(environment) in cont_extex_envs:
                stages_a = stages[0]
                stages_e = stages[1]

                p_gain = mp['l_e'] / (environment.physical_system.tau * a) / u_e_lim * i_e_lim
                i_gain = p_gain / (environment.physical_system.tau * a ** 2)

                stages_e[0]['p_gain'] = p_gain if 'p_gain' not in stages_e[0].keys() else stages_e[0]['p_gain']
                stages_e[0]['i_gain'] = i_gain if 'i_gain' not in stages_e[0].keys() else stages_e[0]['i_gain']

                if stages_e[0]['controller_type'] == PID_Controller:
                    d_gain = p_gain * environment.physical_system.tau
                    stages_e[0]['d_gain'] = d_gain if 'd_gain' not in stages_e[0].keys() else stages_e[0]['d_gain']
            elif type(environment) in finite_extex_envs:
                stages_a = stages[0]
                stages_e = stages[1]
            else:
                stages_a = stages
                stages_e = False

            if _controllers[controller_type][0] == ContinuousActionController:
                if 'i' in ref_states or 'i_a' in ref_states or 'torque' in ref_states:
                    p_gain = mp['l'] / (environment.physical_system.tau * a) / u_a_lim * i_a_lim
                    i_gain = p_gain / (environment.physical_system.tau * a ** 2)

                    stages_a[0]['p_gain'] = p_gain if 'p_gain' not in stages_a[0].keys() else stages_a[0]['p_gain']
                    stages_a[0]['i_gain'] = i_gain if 'i_gain' not in stages_a[0].keys() else stages_a[0]['i_gain']

                    if _controllers[controller_type][2] == PID_Controller:
                        d_gain = p_gain * environment.physical_system.tau
                        stages_a[0]['d_gain'] = d_gain if 'd_gain' not in stages_a[0].keys() else stages_a[0]['d_gain']

                elif 'omega' in ref_states:
                    p_gain = environment.physical_system.mechanical_load.j_total * mp['r_a'] ** 2 / (
                            a * mp['l']) / u_a_lim * omega_lim
                    i_gain = p_gain / (a * mp['l'])

                    stages_a[0]['p_gain'] = p_gain if 'p_gain' not in stages_a[0].keys() else stages_a[0]['p_gain']
                    stages_a[0]['i_gain'] = i_gain if 'i_gain' not in stages_a[0].keys() else stages_a[0]['i_gain']

                    if _controllers[controller_type][2] == PID_Controller:
                        d_gain = p_gain * environment.physical_system.tau
                        stages_a[0]['d_gain'] = d_gain if 'd_gain' not in stages_a[0].keys() else stages_a[0]['d_gain']

            elif _controllers[controller_type][0] == Cascaded_Controller:

                for i in range(len(stages)):

                    if _controllers[stages_a[i]['controller_type']][1] == ContinuousController:

                        if i == 0:
                            p_gain = mp['l'] / (environment.physical_system.tau * a) / u_a_lim * i_a_lim
                            i_gain = p_gain / (environment.physical_system.tau * a ** 2)

                            if _controllers[stages_a[i]['controller_type']][2] == PID_Controller:
                                d_gain = p_gain * environment.physical_system.tau
                                stages_a[i]['d_gain'] = d_gain if 'd_gain' not in stages_a[i].keys() else stages_a[i][
                                    'd_gain']

                        elif i == 1:
                            p_gain = environment.physical_system.mechanical_load.j_total / (
                                    a * (mp['l'] / mp['r_a'])) / i_a_lim * omega_lim
                            i_gain = p_gain / (a ** 2 * (mp['l'] / mp['r_a']))

                            if _controllers[stages_a[i]['controller_type']][2] == PID_Controller:
                                d_gain = p_gain * environment.physical_system.tau
                                stages_a[i]['d_gain'] = d_gain if 'd_gain' not in stages_a[i].keys() else stages_a[i][
                                    'd_gain']

                        stages_a[i]['p_gain'] = p_gain if 'p_gain' not in stages_a[i].keys() else stages_a[i]['p_gain']
                        stages_a[i]['i_gain'] = i_gain if 'i_gain' not in stages_a[i].keys() else stages_a[i]['i_gain']
                if not stages_e:
                    stages = stages_a
                else:
                    stages = [stages_a, stages_e]

            elif _controllers[controller_type][0] == FOC_Controller:
                stage_d = stages[0][0]
                stage_q = stages[1][0]
                if 'i_sq' in ref_states and _controllers[stage_q['controller_type']][1] == ContinuousController:
                    p_gain_d = mp['l_d'] / (1.5 * environment.physical_system.tau * a) / u_sd_lim * i_sd_lim
                    i_gain_d = p_gain_d / (1.5 * environment.physical_system.tau * a ** 2)

                    p_gain_q = mp['l_q'] / (1.5 * environment.physical_system.tau * a) / u_sq_lim * i_sq_lim
                    i_gain_q = p_gain_q / (1.5 * environment.physical_system.tau * a ** 2)

                    stage_d['p_gain'] = p_gain_d if 'p_gain' not in stage_d.keys() else stage_d['p_gain']
                    stage_d['i_gain'] = i_gain_d if 'i_gain' not in stage_d.keys() else stage_d['i_gain']

                    stage_q['p_gain'] = p_gain_q if 'p_gain' not in stage_q.keys() else stage_q['p_gain']
                    stage_q['i_gain'] = i_gain_q if 'i_gain' not in stage_q.keys() else stage_q['i_gain']

                    if _controllers[stage_d['controller_type']][2] == PID_Controller:
                        d_gain_d = p_gain_d * environment.physical_system.tau
                        stage_d['d_gain'] = d_gain_d if 'd_gain' not in stage_d.keys() else stage_d['d_gain']

                    if _controllers[stage_q['controller_type']][2] == PID_Controller:
                        d_gain_q = p_gain_q * environment.physical_system.tau
                        stage_q['d_gain'] = d_gain_q if 'd_gain' not in stage_q.keys() else stage_q['d_gain']
                    stages = [[stage_d], [stage_q]]

            elif _controllers[controller_type][0] == Cascaded_FOC_Controller:
                if type(environment.action_space) == Box:
                    stage_d = stages[0][0]
                    stage_q = stages[1]

                    p_gain_d = mp['l_d'] / (1.5 * environment.physical_system.tau * a) / u_sd_lim * i_sd_lim
                    i_gain_d = p_gain_d / (1.5 * environment.physical_system.tau * a ** 2)

                    p_gain_q = mp['l_q'] / (1.5 * environment.physical_system.tau * a) / u_sq_lim * i_sq_lim
                    i_gain_q = p_gain_q / (1.5 * environment.physical_system.tau * a ** 2)

                    stage_d['p_gain'] = p_gain_d if 'p_gain' not in stage_d.keys() else stage_d['p_gain']
                    stage_d['i_gain'] = i_gain_d if 'i_gain' not in stage_d.keys() else stage_d['i_gain']

                    stage_q[0]['p_gain'] = p_gain_q if 'p_gain' not in stage_q[0].keys() else stage_q[0]['p_gain']
                    stage_q[0]['i_gain'] = i_gain_q if 'i_gain' not in stage_q[0].keys() else stage_q[0]['i_gain']

                    if _controllers[stage_d['controller_type']][2] == PID_Controller:
                        d_gain_d = p_gain_d * environment.physical_system.tau
                        stage_d['d_gain'] = d_gain_d if 'd_gain' not in stage_d.keys() else stage_d['d_gain']

                    if _controllers[stage_q[0]['controller_type']][2] == PID_Controller:
                        d_gain_q = p_gain_q * environment.physical_system.tau
                        stage_q[0]['d_gain'] = d_gain_q if 'd_gain' not in stage_q[0].keys() else stage_q[0]['d_gain']

                    if ('omega' in ref_states or 'torque' in ref_states) and _controllers[stage_q[1]['controller_type']][1] == ContinuousController:
                        p_gain = environment.physical_system.mechanical_load.j_total / (1.5 * a ** 2 * mp['p'] * np.abs(mp['l_d'] - mp['l_q'])) / i_sq_lim * omega_lim
                        i_gain = p_gain / (1.5 * environment.physical_system.tau * a)

                        stage_q[1]['p_gain'] = p_gain if 'p_gain' not in stage_q[1].keys() else stage_q[1]['p_gain']
                        stage_q[1]['i_gain'] = i_gain if 'i_gain' not in stage_q[1].keys() else stage_q[1]['i_gain']

                        if _controllers[stage_q[1]['controller_type']][2] == PID_Controller:
                            d_gain = p_gain * environment.physical_system.tau
                            stage_q[1]['d_gain'] = d_gain if 'd_gain' not in stage_q[1].keys() else stage_q[1][
                                'd_gain']

                    stages = [[stage_d], stage_q]


                else:
                    if ('omega' in ref_states or 'torque' in ref_states) and _controllers[stages[3][0]['controller_type']][1] == ContinuousController:
                        p_gain = environment.physical_system.mechanical_load.j_total / (
                                    1.5 * a ** 2 * mp['p'] * np.abs(mp['l_d'] - mp['l_q'])) / i_sq_lim * omega_lim
                        i_gain = p_gain / (1.5 * environment.physical_system.tau * a)

                        stages[3][0]['p_gain'] = p_gain if 'p_gain' not in stages[3][0].keys() else stages[3][0]['p_gain']
                        stages[3][0]['i_gain'] = i_gain if 'i_gain' not in stages[3][0].keys() else stages[3][0]['i_gain']

                        if _controllers[stages[3][0]['controller_type']][2] == PID_Controller:
                            d_gain = p_gain * environment.physical_system.tau
                            stages[3][0]['d_gain'] = d_gain if 'd_gain' not in stages[3][0].keys() else stages[3][0][
                                'd_gain']

        return stages


class ContinuousActionController(Controller):
    def __init__(self, environment, stages, ref_states, **controller_kwargs):
        assert type(environment.action_space) is Box and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Continuous Action Controller'
        self.action_space = environment.action_space
        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]
        self.omega_idx = environment.state_names.index('omega')
        self.action = np.zeros(self.action_space.shape[0])
        self.control_e = type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor
        mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_e = None if 'psi_e' not in mp.keys() else mp['psi_e']
        self.l_e = None if 'l_e_prime' not in mp.keys() else mp['l_e_prime']

        if self.control_e:
            assert len(stages) == 2, 'Controller design is not completely'
            self.ref_e_idx = False if 'i_e' not in ref_states else np.where(ref_states == 'i_e')[0][0]
            self.ref_e = 0.1 if 'ref_e' not in controller_kwargs.keys() else controller_kwargs['ref_e']
            self.controller_e = _controllers[stages[1][0]['controller_type']][1].make(environment, stages[1][0],
                                                                                      **controller_kwargs)
            self.controller = _controllers[stages[0][0]['controller_type']][1].make(environment, stages[0][0],
                                                                                    **controller_kwargs)
        else:
            assert len(ref_states) <= 1, 'Too many referenced states'
            self.controller = _controllers[stages[0]['controller_type']][1].make(environment, stages[0],
                                                                                 **controller_kwargs)

    def control(self, state, reference):
        self.action[0] = self.controller.control(state[self.ref_state_idx], reference[self.ref_idx]) + self.feedforward(
            state)
        if self.action_space.low[0] <= self.action[0] <= self.action_space.high[0]:
            self.controller.integrate(state[self.ref_state_idx], reference[self.ref_idx])
        if self.control_e:
            ref_e = self.ref_e if not self.ref_e_idx else reference[self.ref_e_idx]
            self.action[1] = self.controller_e.control(state[self.i_idx], ref_e)
            if self.action_space.low[1] <= self.action[1] <= self.action_space.high[1]:
                self.controller_e.integrate(state[self.i_idx], ref_e)
        return np.clip(self.action, self.action_space.low, self.action_space.high)

    def plot(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        self.controller.reset()
        if self.control_e:
            self.controller_e.reset()

    def feedforward(self, state):
        psi_e = self.psi_e or self.l_e * state[self.i_idx] * self.limit[self.i_idx]
        return (state[self.omega_idx] * self.limit[self.omega_idx] * psi_e) / self.limit[self.u_idx]


class DiscreteActionController(Controller):
    def __init__(self, environment, stages, ref_states, **controller_kwargs):
        assert type(environment.action_space) in [Discrete, MultiDiscrete] and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Discrete Action Controller'
        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.control_e = type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor

        if self.control_e:
            assert len(stages) == 2, 'Controller design is not completely'
            self.ref_e_idx = False if 'i_e' not in ref_states else np.where(ref_states == 'i_e')[0][0]
            self.ref_e = 0.1 if 'ref_e' not in controller_kwargs.keys() else controller_kwargs['ref_e']
            self.controller_e = _controllers[stages[1][0]['controller_type']][1].make(environment, stages[1][0],
                                                                                      control_e=True,
                                                                                      **controller_kwargs)
            self.controller = _controllers[stages[0][0]['controller_type']][1].make(environment, stages[0][0],
                                                                                    **controller_kwargs)
        else:
            assert len(ref_states) <= 1, 'Too many referenced states'
            self.controller = _controllers[stages[0]['controller_type']][1].make(environment, stages[0],
                                                                                 **controller_kwargs)

    def control(self, state, reference):
        if self.control_e:
            ref_e = self.ref_e if not self.ref_e_idx else reference[self.ref_e_idx]
            return [self.controller.control(state[self.ref_state_idx], reference[self.ref_idx]),
                    self.controller_e.control(state[self.i_idx], ref_e)]
        else:
            return self.controller.control(state[self.ref_state_idx], reference[self.ref_idx])

    def plot(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        self.controller.reset()
        if self.control_e:
            self.control_e.reset()


class Cascaded_Controller(Controller):
    def __init__(self, environment, stages, ref_states, color='g', **controller_kwargs):

        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space

        self.i_e_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.i_a_idx = environment.physical_system.CURRENTS_IDX[0]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.omega_idx = environment.state_names.index('omega')
        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = [self.i_a_idx, environment.state_names.index(ref_states[self.ref_idx])]

        self.limit = environment.physical_system.limits[environment.state_filter]
        self.control_e = type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor
        mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_e = None if 'psi_e' not in mp.keys() else mp['psi_e']
        self.l_e = None if 'l_e_prime' not in mp.keys() else mp['l_e_prime']

        if self.control_e:
            assert len(stages) == 2, 'Controller design is not completely'
            self.ref_e_idx = False if 'i_e' not in ref_states else np.where(ref_states == 'i_e')[0][0]
            self.ref_e = 0.1 if 'ref_e' not in controller_kwargs.keys() else controller_kwargs['ref_e']
            self.controller_e = _controllers[stages[1][0]['controller_type']][1].make(environment, stages[1][0],
                                                                                      control_e=True,
                                                                                      **controller_kwargs)
            stages = stages[0]
        else:
            assert len(ref_states) <= 1, 'Too many referenced states'

        self.stage_type = [_controllers[stage['controller_type']][1] == ContinuousController for stage in stages]
        self.controller_stages = [
            _controllers[stage['controller_type']][1].make(environment, stage, cascaded=stages.index(stage) != 0) for
            stage in stages]

        assert type(self.action_space) is Box or not self.stage_type[0], 'No suitable inner controller'
        assert type(self.action_space) in [Discrete, MultiDiscrete] or self.stage_type[
            0], 'No suitable inner controller'

        self.ref = np.zeros(len(self.controller_stages))
        self.color = color

    def control(self, state, reference):
        self.ref[-1] = reference[self.ref_idx]
        for i in range(len(self.controller_stages) - 1, 0, -1):
            self.ref[i - 1] = self.controller_stages[i].control(state[self.ref_state_idx[i]], self.ref[i])
            if (0.85 * self.state_space.low[self.ref_state_idx[i - 1]] <= self.ref[i - 1] <= 0.85 *
                self.state_space.high[
                    self.ref_state_idx[i - 1]]) and self.stage_type[i]:
                self.controller_stages[i].integrate(state[self.ref_state_idx[i]], reference[0])
            elif self.stage_type[i]:
                self.ref[i - 1] = np.clip(self.ref[i - 1], 0.85 * self.state_space.low[self.ref_state_idx[i - 1]],
                                          0.85 * self.state_space.high[self.ref_state_idx[i - 1]])

        action = self.controller_stages[0].control(state[self.ref_state_idx[0]], self.ref[0])
        if self.stage_type[0]:
            action += self.feedforward(state)

            if self.action_space.low[0] <= action <= self.action_space.high[0]:
                self.controller_stages[0].integrate(state[self.ref_state_idx[0]], self.ref[0])
                action = [action]
            else:
                action = np.clip([action], self.action_space.low[0], self.action_space.high[0])

        if self.control_e:
            ref_e = self.ref_e if not self.ref_e_idx else reference[self.ref_e_idx]
            action_u_e = self.controller_e.control(state[self.i_e_idx], ref_e)
            if self.stage_type[0]:
                action = np.append(action, action_u_e)
                if self.action_space.low[1] <= action[1] <= self.action_space.high[1]:
                    self.controller_e.integrate(state[self.i_e_idx], ref_e)
                action = np.clip(action, self.action_space.low, self.action_space.high)
            else:
                action = np.array([action, action_u_e], dtype='object')

        return action

    def feedforward(self, state):
        psi_e = max(self.psi_e or self.l_e * state[self.i_e_idx] * self.limit[self.i_e_idx], 1e-6)
        return (state[self.omega_idx] * self.limit[self.omega_idx] * psi_e) / self.limit[self.u_idx]

    def plot(self):
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1], color=self.color)

    def reset(self):
        for controller in self.controller_stages:
            controller.reset()
        if self.control_e:
            self.controller_e.reset()


class FOC_Controller(Controller):
    def __init__(self, environment, stages, ref_states, **controller_kwargs):
        assert type(environment.physical_system) is SynchronousMotorSystem, 'No suitable Environment for FOC Controller'

        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self.backward_transformation = (lambda quantities, eps: t32(q(quantities[::-1], eps)))

        self.tau = environment.physical_system.tau

        self.ref_d_idx = np.where(ref_states == 'i_sd')[0][0]
        self.ref_q_idx = np.where(ref_states == 'i_sq')[0][0]

        self.d_idx = environment.state_names.index(ref_states[self.ref_d_idx])
        self.q_idx = environment.state_names.index(ref_states[self.ref_q_idx])

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
        self.psi_p = 0 if 'psi_p' not in self.mp.keys() else self.mp['psi_p']
        self.dead_time = 1.5 if environment.physical_system.converter._dead_time else 0.5
        self.control_type = type(self.action_space) == Box

        if self.control_type:
            assert len(stages) == 2, 'Number of stages not correct'
            self.decoupling = True if 'decoupling' not in controller_kwargs else controller_kwargs['decoupling']
            [self.u_sq_0, self.u_sd_0] = [0, 0]

            self.d_controller = _controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[0][0], **controller_kwargs)
            self.q_controller = _controllers[stages[1][0]['controller_type']][1].make(
                environment, stages[1][0], **controller_kwargs)

        else:
            assert len(stages) == 3, 'Number of stages not correct'
            self.abc_controller = [_controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[i][0], **controller_kwargs) for i in range(3)]
            self.i_abc_idx = [environment.state_names.index(state) for state in ['i_a', 'i_b', 'i_c']]

    def control(self, state, reference):
        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + self.dead_time * self.tau * \
                    state[self.omega_idx] * self.limit[self.omega_idx] * self.mp['p']
        if self.control_type:
            if self.decoupling:
                self.u_sd_0 = -state[self.omega_idx] * self.mp['p'] * self.mp['l_q'] * state[self.i_sq_idx] * self.limit[
                    self.i_sq_idx] / self.limit[self.u_sd_idx] * self.limit[self.omega_idx]
                self.u_sq_0 = state[self.omega_idx] * self.mp['p'] * (
                        state[self.i_sd_idx] * self.mp['l_d'] * self.limit[self.u_sd_idx] + self.psi_p) / self.limit[
                             self.u_sq_idx] * self.limit[self.omega_idx]

            u_sd = self.d_controller.control(state[self.d_idx], reference[self.ref_d_idx]) + self.u_sd_0
            u_sq = self.q_controller.control(state[self.q_idx], reference[self.ref_q_idx]) + self.u_sq_0

            action_temp = self.backward_transformation((u_sq, u_sd), epsilon_d)
            action_temp = action_temp - 0.5 * (max(action_temp) + min(action_temp))

            action = np.clip(action_temp, self.action_space.low[0], self.action_space.high[0])
            if action.all() == action_temp.all():
                self.d_controller.integrate(state[self.d_idx], reference[self.ref_d_idx])
                self.q_controller.integrate(state[self.q_idx], reference[self.ref_q_idx])

        else:
            ref_abc = self.backward_transformation((reference[self.ref_q_idx], reference[self.ref_d_idx]), epsilon_d)
            action = 0
            for i in range(3):
                action += (2 ** (2 - i)) * self.abc_controller[i].control(state[self.i_abc_idx[i]], ref_abc[i])

        return action

    def plot(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        None


class Cascaded_FOC_Controller(Controller):
    def __init__(self, environment, stages, ref_states, color='g', **controller_kwargs):
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self.backward_transformation = (lambda quantities, eps: t32(q(quantities[::-1], eps)))

        self.tau = environment.physical_system.tau

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

        self.ref_d_idx = np.where(ref_states == 'i_sd')[0][0]
        self.ref_idx = np.where(ref_states != 'i_sd')[0][0]
        self.ref_state_idx = [self.i_sq_idx, environment.state_names.index(ref_states[self.ref_idx])]
        cont_pmsm_envs = (
            envs.DqContCurrentControlPermanentMagnetSynchronousMotorEnv,
            envs.DqContTorqueControlSquirrelCageInductionMotorEnv,
            envs.DqContSpeedControlPermanentMagnetSynchronousMotorEnv,
            envs.AbcContCurrentControlPermanentMagnetSynchronousMotorEnv,
            envs.AbcContTorqueControlPermanentMagnetSynchronousMotorEnv,
            envs.AbcContSpeedControlPermanentMagnetSynchronousMotorEnv
        )
        self.omega_control = 'omega' in ref_states and type(environment) in cont_pmsm_envs
        self.controller_type = type(self.action_space) == Box

        self.limit = environment.physical_system.limits
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_p = 0 if 'psi_p' not in self.mp.keys() else self.mp['psi_p']
        self.dead_time = 1.5 if environment.physical_system.converter._dead_time else 0.5
        self.decoupling = True if 'decoupling' not in controller_kwargs else controller_kwargs['decoupling']

        if self.controller_type:
            assert len(stages) == 2, 'Number of stages not correct'
            self.d_controller = _controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[0][0], **controller_kwargs)
            self.q_controller = _controllers[stages[1][0]['controller_type']][1].make(
                environment, stages[1][0], **controller_kwargs)
            self.overlayed_controller = [_controllers[stages[1][i]['controller_type']][1].make(
                environment, stages[1][i], cascaded=True, **controller_kwargs) for i in range(1, len(stages[1]))]
            self.overlayed_type = [_controllers[stages[1][i]['controller_type']][1] == ContinuousController for i in range(1, len(stages[1]))]
            self.control_type = type(environment.action_space) == Box
            [self.u_sq_0, self.u_sd_0] = [0, 0]

        else:
            assert len(stages) == 4, 'Number of stages not correct'
            self.overlayed_controller = [_controllers[stages[3][i]['controller_type']][1].make(
                environment, stages[3][i], cascaded=True, **controller_kwargs) for i in range(len(stages[3]))]
            self.overlayed_type = [_controllers[stages[3][i]['controller_type']][1] == ContinuousController for i in
                                   range(len(stages[3]))]
            self.abc_controller = [_controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[i][0], **controller_kwargs) for i in range(3)]
            self.i_abc_idx = [environment.state_names.index(state) for state in ['i_a', 'i_b', 'i_c']]

        self.ref = np.zeros(len(self.overlayed_controller) + 1)
        self.color = color

    def control(self, state, reference):
        self.ref[-1] = reference[self.ref_idx]
        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + self.dead_time * self.tau * state[self.omega_idx] * \
                    self.limit[self.omega_idx] * self.mp['p']

        for i in range(len(self.overlayed_controller) - 1, -1, -1):
            self.ref[i] = self.overlayed_controller[i].control(state[self.ref_state_idx[i + 1]], self.ref[i + 1])
            if (0.85 * self.state_space.low[self.ref_state_idx[i]] <= self.ref[i] <= 0.85 *
                    self.state_space.high[self.ref_state_idx[i]]) and self.overlayed_type[i]:
                self.overlayed_controller[i].integrate(state[self.ref_state_idx[i + 1]], self.ref[i + 1])
            else:
                self.ref[i] = np.clip(self.ref[i], 0.85 * self.state_space.low[self.ref_state_idx[i]],
                                      0.85 * self.state_space.high[self.ref_state_idx[i]])
        if self.omega_control:
            self.ref[0] = 2 / (3 * self.mp['p'] * self.mp['psi_p']) * self.ref[0]
            self.ref[0] = np.clip(self.ref[0], 0.85 * self.state_space.low[self.ref_state_idx[0]],
                                  0.85 * self.state_space.high[self.ref_state_idx[0]])

        if self.controller_type:
            if self.decoupling:
                self.u_sd_0 = -state[self.omega_idx] * self.mp['p'] * self.mp['l_q'] * state[self.i_sq_idx] * self.limit[
                    self.i_sq_idx] / self.limit[self.u_sd_idx] * self.limit[self.omega_idx]
                self.u_sq_0 = state[self.omega_idx] * self.mp['p'] * (
                        state[self.i_sd_idx] * self.mp['l_d'] * self.limit[self.u_sd_idx] + self.psi_p) / self.limit[
                         self.u_sq_idx] * self.limit[self.omega_idx]

            u_sd = self.d_controller.control(state[self.i_sd_idx], reference[self.ref_d_idx]) + self.u_sd_0
            u_sq = self.q_controller.control(state[self.i_sq_idx], self.ref[0]) + self.u_sq_0

            action_temp = self.backward_transformation((u_sq, u_sd), epsilon_d)
            action_temp = action_temp - 0.5 * (max(action_temp) + min(action_temp))

            action = np.clip(action_temp, self.action_space.low[0], self.action_space.high[0])
            if action.all() == action_temp.all():
                self.d_controller.integrate(state[self.i_sd_idx], reference[self.ref_d_idx])
                self.q_controller.integrate(state[self.i_sq_idx], self.ref[0])

        else:
            ref_abc = self.backward_transformation((self.ref[0], reference[self.ref_d_idx]), epsilon_d)
            action = 0
            for i in range(3):
                action += (2 ** (2 - i)) * self.abc_controller[i].control(state[self.i_abc_idx[i]], ref_abc[i])

        return action

    def plot(self):
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1], color=self.color)

    def reset(self):
        self.overlayed_controller.reset()
        if self.controller_type:
            self.d_controller.reset()
            for q_controller in self.q_controller:
                q_controller.reset()
        else:
            for abc_controller in self.abc_controller:
                abc_controller.reset()



class ContinuousController:
    @classmethod
    def make(cls, environment, stage, **controller_kwargs):
        controller = _controllers[stage['controller_type']][2](environment, param_dict=stage, **controller_kwargs)
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
    def __init__(self, environment, p_gain=5, i_gain=5, param_dict={}, **controller_kwargs):
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
    def __init__(self, environment, p_gain=5, i_gain=5, d_gain=0.005, param_dict={}, **controller_kwargs):
        p_gain = param_dict['p_gain'] if 'p_gain' in param_dict.keys() else p_gain
        i_gain = param_dict['i_gain'] if 'i_gain' in param_dict.keys() else i_gain
        d_gain = param_dict['d_gain'] if 'd_gain' in param_dict.keys() else d_gain

        PI_Controller.__init__(self, environment, p_gain, i_gain)
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
    def make(cls, environment, stage, **controller_kwargs):
        if type(environment.action_space) == Discrete:
            action_space_n = environment.action_space.n
        elif type(environment.action_space) == MultiDiscrete:
            action_space_n = environment.action_space.nvec[0]
        else:
            action_space_n = 3

        controller = _controllers[stage['controller_type']][2](environment, action_space=action_space_n,
                                                               param_dict=stage, **controller_kwargs)
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass


class OnOff_Controller(DiscreteController):
    def __init__(self, environment, action_space, hysteresis=0.02, param_dict={}, cascaded=False, control_e=False,
                 **controller_kwargs):
        self.hysteresis = hysteresis if 'hysteresis' not in param_dict.keys() else param_dict['hysteresis']
        self.switch_on_level = 1

        self.switch_off_level = 2 if action_space in [3, 4] and not control_e else 0
        if cascaded:
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
    def __init__(self, environment, action_space, switch_to_positive_level=0.02, switch_to_negative_level=0.02,
                 switch_to_neutral_from_positive=0.01, switch_to_neutral_from_negative=0.01, param_dict={},
                 cascaded=False, control_e=False, **controller_kwargs):

        self.pos = switch_to_positive_level if 'switch_to_positive_level' not in param_dict.keys() else param_dict[
            'switch_to_positive_level']
        self.neg = switch_to_negative_level if 'switch_to_negative_level' not in param_dict.keys() else param_dict[
            'switch_to_negative_level']
        self.neutral_from_pos = switch_to_neutral_from_positive if 'switch_to_neutral_from_positive' not in param_dict.keys() else \
            param_dict['switch_to_neutral_from_positive']
        self.neutral_from_neg = switch_to_neutral_from_negative if 'switch_to_neutral_from_negative' not in param_dict.keys() else \
            param_dict['switch_to_neutral_from_negative']

        self.negative = 2 if action_space in [3, 4, 8] and not control_e else 0
        if cascaded:
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
    'cascaded_foc_controller': [Cascaded_FOC_Controller],
    #    'foc_rotor_flux_observer': FOC_Rotor_Flux_Observer
}
