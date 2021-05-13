from gym.spaces import Discrete, Box, MultiDiscrete
from gym_electric_motor.physical_systems import SynchronousMotorSystem, DcMotorSystem, DcSeriesMotor, \
    DcExternallyExcitedMotor
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, SwitchedReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor import envs
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class Controller:

    @classmethod
    def make(cls, environment, stages=None, **controller_kwargs):

        controller_kwargs = cls.reference_states(environment, **controller_kwargs)
        cls.visualization, controller_kwargs = cls.get_visualization(environment, **controller_kwargs)

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

    def set_ref(self):
        pass

    def reset(self):
        pass

    def plot(self, external_reference_plots, state_names):
        if self.visualization:
            external_refs = self.set_ref()
            external_ref_plots = list(external_reference_plots)
            ref_state_idxs = external_refs['ref_state']
            plot_state_idxs = [
                list(state_names).index(external_ref_plot.state) for external_ref_plot in external_reference_plots
            ]
            ref_values = external_refs['ref_value']
            for ref_state_idx, ref_value in zip(ref_state_idxs, ref_values):
                try:
                    plot_idx = plot_state_idxs.index(ref_state_idx)
                except ValueError:
                    pass
                else:
                    external_ref_plots[plot_idx].external_reference(ref_value)

    @staticmethod
    def get_visualization(environment, **controller_kwargs):
        for visualization in environment._visualizations:
            if type(visualization) == MotorDashboard:
                controller_kwargs['update_interval'] = visualization._update_interval
                return True, controller_kwargs
        return False, controller_kwargs

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
            torque_lim = limits[environment.state_names.index('torque')]

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
                            t_n = p_gain / i_gain
                            p_gain = environment.physical_system.mechanical_load.j_total / (
                                    4 * t_n) / i_a_lim * omega_lim
                            i_gain = p_gain / (4 * t_n)

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

                    if 'omega' in ref_states and _controllers[stage_q[1]['controller_type']][1] == ContinuousController:
                        t_n = p_gain_d / i_gain_d
                        p_gain = environment.physical_system.mechanical_load.j_total / (4 * t_n) / torque_lim * omega_lim
                        i_gain = p_gain / (4 * t_n)


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
    def __init__(self, environment, stages, ref_states, external_ref_plots=[], **controller_kwargs):
        assert type(environment.action_space) is Box and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Continuous Action Controller'
        self.action_space = environment.action_space
        self.state_names = environment.state_names
        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]
        self.nominal_values = environment.physical_system.nominal_state[environment.state_filter]
        self.omega_idx = environment.state_names.index('omega')
        self.action = np.zeros(self.action_space.shape[0])
        self.control_e = 'i_e' in ref_states
        mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_e = None if 'psi_e' not in mp.keys() else mp['psi_e']
        self.l_e = None if 'l_e_prime' not in mp.keys() else mp['l_e_prime']
        self.external_ref_plots = external_ref_plots

        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

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
        self.plot(self.external_ref_plots, self.state_names)
        return np.clip(self.action, self.action_space.low, self.action_space.high)

    def set_ref(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        self.controller.reset()
        if self.control_e:
            self.controller_e.reset()

    def feedforward(self, state):
        psi_e = self.psi_e or self.l_e * state[self.i_idx] * self.nominal_values[self.i_idx]
        return (state[self.omega_idx] * self.nominal_values[self.omega_idx] * psi_e) / self.nominal_values[self.u_idx]


class DiscreteActionController(Controller):
    def __init__(self, environment, stages, ref_states, external_ref_plots=[], **controller_kwargs):

        assert type(environment.action_space) in [Discrete, MultiDiscrete] and type(
            environment.physical_system) is DcMotorSystem, 'No suitable action space for Discrete Action Controller'

        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.control_e = type(environment.physical_system.electrical_motor) == DcExternallyExcitedMotor
        self.state_names = environment.state_names

        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

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
        self.plot(self.external_ref_plots, self.state_names)
        if self.control_e:
            ref_e = self.ref_e if not self.ref_e_idx else reference[self.ref_e_idx]
            return [self.controller.control(state[self.ref_state_idx], reference[self.ref_idx]),
                    self.controller_e.control(state[self.i_idx], ref_e)]
        else:
            return self.controller.control(state[self.ref_state_idx], reference[self.ref_idx])

    def set_ref(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        self.controller.reset()
        if self.control_e:
            self.control_e.reset()


class Cascaded_Controller(Controller):
    def __init__(self, environment, stages, ref_states, external_ref_plots=[], **controller_kwargs):

        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space
        self.state_names = environment.state_names

        self.i_e_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.i_a_idx = environment.physical_system.CURRENTS_IDX[0]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.omega_idx = environment.state_names.index('omega')
        self.ref_idx = np.where(ref_states != 'i_e')[0][0]
        self.ref_state_idx = [self.i_a_idx, environment.state_names.index(ref_states[self.ref_idx])]

        self.limit = environment.physical_system.limits[environment.state_filter]
        self.nominal_values = environment.physical_system.nominal_state[environment.state_filter]
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

        self.external_ref_plots = external_ref_plots
        internal_refs = np.array([environment.state_names[i] for i in self.ref_state_idx])
        ref_states_plotted = np.unique(np.append(ref_states, internal_refs))
        for external_plots in self.external_ref_plots:
            external_plots.set_reference(ref_states_plotted)

        assert type(self.action_space) is Box or not self.stage_type[0], 'No suitable inner controller'
        assert type(self.action_space) in [Discrete, MultiDiscrete] or self.stage_type[
            0], 'No suitable inner controller'

        self.ref = np.zeros(len(self.controller_stages))

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
        self.plot(self.external_ref_plots, self.state_names)
        return action

    def feedforward(self, state):
        psi_e = max(self.psi_e or self.l_e * state[self.i_e_idx] * self.nominal_values[self.i_e_idx], 1e-6)
        return (state[self.omega_idx] * self.nominal_values[self.omega_idx] * psi_e) / self.nominal_values[self.u_idx]

    def set_ref(self):
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1])

    def reset(self):
        for controller in self.controller_stages:
            controller.reset()
        if self.control_e:
            self.controller_e.reset()


class FOC_Controller(Controller):
    def __init__(self, environment, stages, ref_states, external_ref_plots=[], **controller_kwargs):
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
        self.state_names = environment.state_names

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
        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

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

        self.plot(self.external_ref_plots, self.state_names)
        return action

    def set_ref(self):
        return dict(ref_state=[], ref_value=[])

    def reset(self):
        self.d_controller.reset()
        self.q_controller.reset()


class Cascaded_FOC_Controller(Controller):
    def __init__(self, environment, stages, ref_states, external_ref_plots=[], plot_torque=True, plot_modulation=False, update_interval=1000, **controller_kwargs):
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self.backward_transformation = (lambda quantities, eps: t32(q(quantities[::-1], eps)))
        self.tau = environment.physical_system.tau

        self.action_space = environment.action_space
        self.state_space = environment.physical_system.state_space
        self.state_names = environment.state_names

        self.i_sd_idx = environment.state_names.index('i_sd')
        self.i_sq_idx = environment.state_names.index('i_sq')
        self.u_sd_idx = environment.state_names.index('u_sd')
        self.u_sq_idx = environment.state_names.index('u_sq')
        self.u_a_idx = environment.state_names.index('u_a')
        self.u_b_idx = environment.state_names.index('u_b')
        self.u_c_idx = environment.state_names.index('u_c')
        self.omega_idx = environment.state_names.index('omega')
        self.eps_idx = environment.state_names.index('epsilon')
        self.torque_idx = environment.state_names.index('torque')
        self.external_ref_plots = external_ref_plots

        self.torque_control = 'torque' in ref_states or 'omega' in ref_states
        self.current_control = 'i_sd' in ref_states
        self.omega_control = 'omega' in ref_states
        if 'i_sd' in ref_states:
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
        self.nominal_values = environment.physical_system.nominal_state
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_p = 0 if 'psi_p' not in self.mp.keys() else self.mp['psi_p']
        self.dead_time = 1.5 if environment.physical_system.converter._dead_time else 0.5
        self.decoupling = True if 'decoupling' not in controller_kwargs else controller_kwargs['decoupling']

        self.ref_state_idx = [self.i_sq_idx, self.i_sd_idx]
        if self.torque_control:
            self.ref_state_idx.append(self.torque_idx)

            self.torque_controller = MTPC(environment, plot_torque, plot_modulation, update_interval)

        if self.omega_control:
            self.ref_state_idx.append(self.omega_idx)

        self.ref_idx = 0

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

        self.ref = np.zeros(len(self.ref_state_idx))
        self.p = [environment.state_names[i] for i in self.ref_state_idx]

        plot_ref = np.append(np.array([environment.state_names[i] for i in self.ref_state_idx]), ref_states)

        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(plot_ref)

    def control(self, state, reference):

        self.ref[-1] = reference[self.ref_idx]
        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + self.dead_time * self.tau * state[self.omega_idx] * \
                    self.limit[self.omega_idx] * self.mp['p']

        if self.omega_control:
            for i in range(len(self.overlayed_controller) + 1, 1, -1):
                self.ref[i] = self.overlayed_controller[i-2].control(state[self.ref_state_idx[i + 1]], self.ref[i + 1])

                if (0.85 * self.state_space.low[self.ref_state_idx[i]] <= self.ref[i] <= 0.85 *
                        self.state_space.high[self.ref_state_idx[i]]) and self.overlayed_type[i-2]:
                    self.overlayed_controller[i-2].integrate(state[self.ref_state_idx[i + 1]], self.ref[i + 1])
                else:
                    self.ref[i] = np.clip(self.ref[i], self.nominal_values[self.ref_state_idx[i]] / self.limit[self.ref_state_idx[i]] * self.state_space.low[self.ref_state_idx[i]],
                                          self.nominal_values[self.ref_state_idx[i]] / self.limit[self.ref_state_idx[i]] * self.state_space.high[self.ref_state_idx[i]])
        if self.torque_control:
            torque = self.ref[2] * self.limit[self.torque_idx]
            self.ref[0], self.ref[1] = self.torque_controller.control(state, torque)

        if self.controller_type:
            if self.decoupling:
                self.u_sd_0 = -state[self.omega_idx] * self.mp['p'] * self.mp['l_q'] * state[self.i_sq_idx] * self.limit[
                    self.i_sq_idx] / self.limit[self.u_sd_idx] * self.limit[self.omega_idx]
                self.u_sq_0 = state[self.omega_idx] * self.mp['p'] * (
                        state[self.i_sd_idx] * self.mp['l_d'] * self.limit[self.u_sd_idx] + self.psi_p) / self.limit[
                         self.u_sq_idx] * self.limit[self.omega_idx]

            if self.torque_control:
                u_sd = self.d_controller.control(state[self.i_sd_idx], self.ref[1]) + self.u_sd_0
            else:
                u_sd = self.d_controller.control(state[self.i_sd_idx], reference[self.ref_d_idx]) + self.u_sd_0

            u_sq = self.q_controller.control(state[self.i_sq_idx], self.ref[0]) + self.u_sq_0

            action_temp = self.backward_transformation((u_sq, u_sd), epsilon_d)
            action_temp = action_temp - 0.5 * (max(action_temp) + min(action_temp))

            action = np.clip(action_temp, self.action_space.low[0], self.action_space.high[0])
            if action.all() == action_temp.all():
                if self.torque_control:
                    self.d_controller.integrate(state[self.i_sd_idx], self.ref[1])
                else:
                    self.d_controller.integrate(state[self.i_sd_idx], reference[self.ref_d_idx])
                self.q_controller.integrate(state[self.i_sq_idx], self.ref[0])

        else:
            if self.torque_control:
                ref_abc = self.backward_transformation((self.ref[0], self.ref[1]), epsilon_d)
            else:
                ref_abc = self.backward_transformation((self.ref[0], reference[self.ref_d_idx]), epsilon_d)
            action = 0
            for i in range(3):
                action += (2 ** (2 - i)) * self.abc_controller[i].control(state[self.i_abc_idx[i]], ref_abc[i])

        self.plot(self.external_ref_plots, self.state_names)
        return action

    def set_ref(self):
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1])

    def reset(self):
        for overlayed_controller in self.overlayed_controller:
            overlayed_controller.reset()
        if self.controller_type:
            self.d_controller.reset()
            self.q_controller.reset()

        else:
            for abc_controller in self.abc_controller:
                abc_controller.reset()

        if self.torque_control:
            self.torque_controller.reset()

class MTPC:
    def __init__(self, environment, plot_torque=True, plot_modulation=False, update_interval=1000):
        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.limit = environment.physical_system.limits
        self.nominal_values = environment.physical_system.nominal_state

        self.q = environment.physical_system.electrical_motor.q
        self.a_max = 2 / np.sqrt(3)
        self.k_ = 0.95
        d = 1.2
        alpha = d / (d - np.sqrt(d ** 2 - 1))
        self.i_gain = 1 / (self.mp['l_q'] / (1.25 * self.mp['r_s'])) * (alpha - 1) / alpha ** 2
        self.integrated = 0
        self.integrated_reset = -0.0005
        self.u_a_idx = environment.state_names.index('u_a')
        self.u_dc = np.sqrt(3) * self.limit[self.u_a_idx]
        self.limited = False
        self.psi_low = -0.04
        self.psi_high = 0.04

        l_d = self.mp['l_d']
        l_q = self.mp['l_q']
        self.p = self.mp['p']
        self.psi_p = self.mp['psi_p']
        self.tau = environment.physical_system.tau

        self.omega_idx = environment.state_names.index('omega')
        self.i_sd_idx = environment.state_names.index('i_sd')
        self.i_sq_idx = environment.state_names.index('i_sq')
        self.u_sd_idx = environment.state_names.index('u_sd')
        self.u_sq_idx = environment.state_names.index('u_sq')
        self.torque_idx = environment.state_names.index('torque')
        self.epsilon_idx = environment.state_names.index('epsilon')

        def mtpc():
            def i_q_(i_d, torque):
                return torque / (i_d * (l_d - l_q) + self.psi_p) / (1.5 * self.p)

            def i_d_(i_q, torque):
                return -np.abs(torque / (1.5 * self.p * (l_d - l_q) * i_q))

            psi_p = 0 if 'psi_p' not in self.mp else self.psi_p
            self.max_torque = max(1.5 * self.p * (psi_p + (l_d - l_q) * (-self.limit[self.i_sd_idx])) * self.limit[self.i_sq_idx], self.limit[self.torque_idx])
            self.t_count_mtpc = 2500
            torque = np.linspace(-self.max_torque, self.max_torque, self.t_count_mtpc)
            characteristic = []

            for t in torque:
                if 'psi_p' in self.mp:
                    if l_d == l_q:
                        i_d = 0
                    else:
                        i_d = np.linspace(-2.5*self.limit[self.i_sd_idx], 0, 2000)
                    i_q = i_q_(i_d, t)
                else:
                    i_q = np.linspace(-2.5*self.limit[self.i_sq_idx], 2.5*self.limit[self.i_sq_idx], 2000)
                    if l_d == l_q:
                        i_d = 0
                    else:
                        i_d = i_d_(i_q, t)

                i = np.power(i_d, 2) + np.power(i_q, 2)
                min_idx = np.where(i == np.amin(i))[0][0]
                if l_d == l_q:
                    i_q_ret = i_q
                    i_d_ret = i_d
                else:
                    i_q_ret = np.sign((l_q - l_d) * t) * np.abs(i_q[min_idx])
                    i_d_ret = i_d[min_idx]
                psi = np.sqrt((self.psi_p + l_d * i_d_ret) ** 2 + (l_q * i_q_ret) ** 2)
                characteristic.append([t, i_d_ret, i_q_ret, psi])
            return np.array(characteristic)

        def mtpf():
            psi_i_d_q = []
            self.psi_max_mtpf = np.sqrt((self.psi_p + l_d * self.nominal_values[self.i_sd_idx]) ** 2 + (l_q * self.nominal_values[self.i_sq_idx]) ** 2)
            self.psi_count_mtpf = 2001
            psi = np.linspace(0, self.psi_max_mtpf, self.psi_count_mtpf)
            i_d = np.linspace(-self.nominal_values[self.i_sd_idx], 0, 1500)
            i_d_best = 0
            i_q_best = 0

            for psi_ in psi:
                if psi_ == 0:
                    i_d_ = -self.psi_p / l_d
                    i_q = 0
                    t = 0
                    psi_i_d_q.append([psi_, t, i_d_, i_q])

                else:
                    if self.psi_p == 0:
                        i_q_best = psi_ / np.sqrt(l_d ** 2 + l_q ** 2)
                        i_d_best = -i_q_best
                        t = 1.5 * self.p * (self.psi_p + (l_d - l_q) * i_d_best) * i_q_best
                    else:
                        i_d_idx = np.where(psi_ ** 2 - np.power(self.psi_p + l_d * i_d, 2) >= 0)
                        i_d_ = i_d[i_d_idx]
                        i_q = np.sqrt(psi_ ** 2 - np.power(self.psi_p + l_d * i_d_, 2)) / l_q
                        i_idx = np.where(np.sqrt(np.power(i_q / self.nominal_values[self.i_sq_idx], 2) + np.power(i_d_ / self.nominal_values[self.i_sd_idx], 2)) <= 1)
                        i_d_ = i_d_[i_idx]
                        i_q = i_q[i_idx]
                        torque = 1.5 * self.p * (self.psi_p + (l_d - l_q) * i_d_) * i_q

                        if np.size(torque) > 0:
                            t = np.amax(torque)
                            i_idx = np.where(torque == t)[0][0]
                            i_d_best = i_d_[i_idx]
                            i_q_best = i_q[i_idx]

                    psi_i_d_q.append([psi_, t, i_d_best, i_q_best])

            psi_i_d_q = np.array(psi_i_d_q)
            psi_i_d_q_neg = np.rot90(np.array([psi_i_d_q[:, 0], -psi_i_d_q[:, 1], psi_i_d_q[:, 2], -psi_i_d_q[:, 3]]))
            psi_i_d_q = np.append(psi_i_d_q_neg, psi_i_d_q, axis=0)
            return np.array(psi_i_d_q)

        self.mtpc = mtpc()
        self.mtpf = mtpf()
        self.psi_t = np.sqrt(
            np.power(self.psi_p + l_d * self.mtpc[:, 1], 2) + np.power(l_q * self.mtpc[:, 2], 2))
        self.psi_t = np.array([self.mtpc[:, 0], self.psi_t])
        self.i_q_max = np.linspace(-self.nominal_values[self.i_sq_idx], self.nominal_values[self.i_sq_idx], 1000)
        self.i_d_max = -np.sqrt(self.nominal_values[self.i_sq_idx] ** 2 - np.power(self.i_q_max, 2))

        i_d, i_q = np.mgrid[-1.3 * self.limit[self.i_sd_idx]:0:700j, -1.3 * self.limit[self.i_sq_idx]:1.3 * self.limit[self.i_sq_idx]:1500j]
        i_d = i_d.flatten()
        i_q = i_q.flatten()
        if l_d != l_q:
            idx = np.where(np.sign(self.psi_p + i_d * l_d) * np.power(self.psi_p + i_d * l_d, 2) + np.power(i_q * l_q, 2) > 0)
        else:
            idx = np.where(self.psi_p + i_d * l_d > 0)
        i_d = i_d[idx]
        i_q = i_q[idx]

        t = self.p * 1.5 * (self.psi_p + (l_d - l_q) * i_d) * i_q
        psi = np.sqrt(np.power(l_d * i_d + self.psi_p, 2) + np.power(l_q * i_q, 2))

        self.t_min = np.amin(t)
        self.t_max = np.amax(t)
        self.t_count = 2001

        self.psi_min = np.amin(psi)
        self.psi_max = np.amax(psi)
        self.psi_count = 601

        self.t_grid, self.psi_grid = np.mgrid[np.amin(t):np.amax(t):np.complex(0, self.t_count), self.psi_min:self.psi_max:np.complex(self.psi_count)]

        self.i_q_inter = griddata((t, psi), i_q, (self.t_grid, self.psi_grid), method='linear')
        self.i_d_inter = griddata((t, psi), i_d, (self.t_grid, self.psi_grid), method='linear')

        self.k = 0
        self.update_interval = update_interval
        self.plot_torque = plot_torque
        self.plot_modulation = plot_modulation
        matplotlib.use('qt5agg')

    def intitialize_torque_plot(self):
        if self.plot_torque:
            plt.ion()
            self.fig_torque = plt.figure('Torque Controller')
            self.i_d_q_characteristic_ = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
            self.psi_plot = plt.subplot2grid((2, 3), (0, 1))
            self.i_d_plot = plt.subplot2grid((2, 3), (0, 2), projection='3d')
            self.torque_plot = plt.subplot2grid((2, 3), (1, 1))
            self.i_q_plot = plt.subplot2grid((2, 3), (1, 2), projection='3d')

            self.i_d_q_characteristic_.set_title('$i_{d,q_{ref}}$')
            self.i_d_q_characteristic_.plot(self.mtpc[:, 1], self.mtpc[:, 2], label='MTPC', c='tab:orange')
            self.i_d_q_characteristic_.plot(self.mtpf[:, 2], self.mtpf[:, 3], label='MTPF', c='tab:green')
            self.i_d_q_characteristic_.plot(self.i_d_max, self.i_q_max, label='$i_{max}$', c='tab:red')
            self.i_d_q_characteristic_.plot([], [], label='$i_{d,q}$', c='tab:blue')
            self.i_d_q_characteristic_.grid(True)
            self.i_d_q_characteristic_.legend(loc=2)
            self.i_d_q_characteristic_.axis('equal')
            self.i_d_q_characteristic_.set_xlabel('$i_d$ / A')
            self.i_d_q_characteristic_.set_ylabel('$i_q$ / A')

            self.psi_plot.set_title('$\Psi^*_{max}(T^*)$')
            self.psi_plot.plot(self.psi_t[0], self.psi_t[1], label='$\Psi^*_{max}(T^*)$', c='tab:orange')
            self.psi_plot.plot([], [], label='$\Psi(T)$', c='tab:blue')
            self.psi_plot.grid(True)
            self.psi_plot.set_xlabel('T / Nm')
            self.psi_plot.set_ylabel('$\Psi$ / Vs')
            self.psi_plot.set_ylim(bottom=0)
            self.psi_plot.legend(loc=2)

            self.torque_plot.set_title('$T_{max}(\Psi)$')
            self.torque_plot.plot(self.mtpf[:, 0], self.mtpf[:, 1], label='$T_{max}(\Psi)$', c='tab:orange')
            self.torque_plot.plot([], [], label='$T(\Psi)$', c='tab:blue')
            self.torque_plot.set_xlabel('$\Psi$ / Vs')
            self.torque_plot.set_ylabel('$T_{max}$ / Nm')
            self.torque_plot.grid(True)
            self.torque_plot.legend(loc=2)


            self.i_q_plot.plot_surface(self.t_grid, self.psi_grid, self.i_q_inter, cmap=cm.jet, linewidth=0, vmin=np.nanmin(self.i_q_inter),
                            vmax=np.nanmax(self.i_q_inter))
            self.i_q_plot.set_ylabel('$\Psi / Vs$')
            self.i_q_plot.set_xlabel('$T / Nm$')
            self.i_q_plot.set_title('$i_q(T, \Psi)$')


            self.i_d_plot.plot_surface(self.t_grid, self.psi_grid, self.i_d_inter, cmap=cm.jet, linewidth=0, vmin=np.nanmin(self.i_d_inter),
                             vmax=np.nanmax(self.i_d_inter))
            self.i_d_plot.set_ylabel('$\Psi / Vs$')
            self.i_d_plot.set_xlabel('$T / Nm$')
            self.i_d_plot.set_title('$i_d(T, \Psi)$')
            self.torque_list = []
            self.psi_list = []
            self.k_list = []
            self.i_d_list = []
            self.i_q_list = []



    def initialize_modulation_plot(self):
        if self.plot_modulation:
            plt.ion()
            self.fig_modulation = plt.figure('Modulation Controller')
            self.a_plot = plt.subplot2grid((1, 2), (0, 0))
            self.psi_delta_plot = plt.subplot2grid((1, 2), (0, 1))

            self.a_plot.set_title('Modulation')
            self.a_plot.axhline(self.k_ * self.a_max, c='tab:orange', label='$a^*$')
            self.a_plot.plot([], [], c='tab:blue', label='a')
            self.a_plot.set_xlabel('t / s')
            self.a_plot.set_ylabel('a')
            self.a_plot.grid(True)
            self.a_plot.set_xlim(0, 1)
            self.a_plot.legend(loc=2)

            self.psi_delta_plot.set_title('$\Psi_{\Delta}$')
            self.psi_delta_plot.axhline(self.psi_low, c='tab:red', linestyle='dashed', label='Limit')
            self.psi_delta_plot.axhline(self.psi_high, c='tab:red', linestyle='dashed')
            self.psi_delta_plot.plot([], [], c='tab:blue', label='$\Psi_{\Delta}$')
            self.psi_delta_plot.set_xlabel('t / s')
            self.psi_delta_plot.set_ylabel('$\Psi_{\Delta} / Vs$')
            self.psi_delta_plot.grid(True)
            self.psi_delta_plot.set_xlim(0, 1)
            self.psi_delta_plot.legend(loc=2)

            self.a_list = []
            self.psi_delta_list = []
            self.k_list_a = []


    def get_t_idx(self, torque):
        return int(round((torque - self.t_min) / (self.t_max - self.t_min) * (self.t_count - 1)))

    def get_psi_idx(self, psi):
        return int(round((psi - self.psi_min) / (self.psi_max - self.psi_min) * (self.psi_count - 1)))

    def get_psi_idx_mtpf(self, psi):
        return int((self.psi_count_mtpf - 1) - round(psi / self.psi_max_mtpf * (self.psi_count_mtpf - 1)))

    def get_t_idx_mtpc(self, torque):
        return int(round((torque + self.max_torque) / (2 * self.max_torque) * (self.t_count_mtpc - 1)))

    def control(self, state, torque):
        psi = self.mtpc[self.get_t_idx_mtpc(torque), 3]
        psi_max_ = self.modulation_control(state)
        psi_max = min(psi, psi_max_)

        psi_max_idx = self.get_psi_idx_mtpf(psi_max)
        t_max = np.abs(self.mtpf[psi_max_idx, 1])
        if np.abs(torque) > t_max:
            i_d = self.mtpf[psi_max_idx, 2]
            i_q = np.sign(torque) * np.abs(self.mtpf[psi_max_idx, 3])
            torque = np.sign(torque) * t_max
        else:
            t_idx = self.get_t_idx(torque)
            psi_idx = self.get_psi_idx(psi_max)
            i_d = self.i_d_inter[t_idx, psi_idx]
            i_q = self.i_q_inter[t_idx, psi_idx]

        if self.plot_torque:
            if self.k == 0:
                self.intitialize_torque_plot()

            self.k_list.append(self.k / 10000)
            self.i_d_list.append(i_d)
            self.i_q_list.append(i_q)
            self.torque_list.append(torque)
            self.psi_list.append(psi_max)

            if self.k % self.update_interval == 0:
                self.psi_plot.scatter(self.torque_list, self.psi_list, c='tab:blue', s=3)
                self.torque_plot.scatter(self.psi_list, self.torque_list, c='tab:blue', s=3)
                self.i_d_q_characteristic_.scatter(self.i_d_list, self.i_q_list, c='tab:blue', s=3)

                self.fig_torque.canvas.draw()
                self.fig_torque.canvas.flush_events()
                self.k_list = []
                self.i_d_list = []
                self.i_q_list = []
                self.torque_list = []
                self.psi_list = []

        i_q = np.clip(i_q, -self.nominal_values[self.i_sq_idx], self.nominal_values[self.i_sq_idx]) / self.limit[self.i_sq_idx]
        i_d = np.clip(i_d, -self.nominal_values[self.i_sd_idx], self.nominal_values[self.i_sd_idx]) / self.limit[self.i_sd_idx]

        self.k += 1

        return i_q, i_d

    def modulation_control(self, state):
        eps = state[self.epsilon_idx] * self.limit[self.epsilon_idx]
        u_sd = state[self.u_sd_idx] * self.limit[self.u_sd_idx]
        u_sq = state[self.u_sq_idx] * self.limit[self.u_sq_idx]
        u_alpha, u_beta = self.q([u_sd, u_sq], eps)
        a = 2 * np.sqrt(u_alpha ** 2 + u_beta ** 2) / self.u_dc

        if a > 1.1 * self.a_max:
            self.integrated = self.integrated_reset

        a_delta = self.k_ * self.a_max - a
        if state[self.omega_idx] > 0.0001:
            psi_max_ = self.u_dc / (
                np.sqrt(3) * self.limit[self.omega_idx] * np.abs(state[self.omega_idx]) * self.p)
            k_i = 2 * self.limit[self.omega_idx] * np.abs(state[self.omega_idx]) * self.p / self.u_dc

        else:
            psi_max_ = self.u_dc / (
                    np.sqrt(3) * self.limit[self.omega_idx] * 0.0001 * self.p)
            k_i = 2 * self.limit[self.omega_idx] * 0.0001 * self.p / self.u_dc
        i_gain = self.i_gain / k_i
        psi_delta = i_gain * (a_delta * self.tau + self.integrated)

        if self.psi_low <= psi_delta <= self.psi_high:
            if self.limited:
                self.integrated = self.integrated_reset
                self.limited = False
            self.integrated += a_delta * self.tau

        else:
            psi_delta = np.clip(psi_delta, self.psi_low, self.psi_high)
            self.limited = True

        psi = psi_max_ + psi_delta
        if self.plot_modulation:
            if self.k == 0:
                self.initialize_modulation_plot()
            self.k_list_a.append(self.k / 10000)
            self.a_list.append(a)
            self.psi_delta_list.append(psi_delta)

            if self.k % self.update_interval == 0:
                    self.a_plot.scatter(self.k_list_a, self.a_list, c='tab:blue', s=3)
                    self.psi_delta_plot.scatter(self.k_list_a, self.psi_delta_list, c='tab:blue', s=3)
                    self.a_plot.set_xlim(max(self.k / 10000, 1) - 1, max(self.k / 10000, 1))
                    self.psi_delta_plot.set_xlim(max(self.k / 10000, 1) - 1, max(self.k / 10000, 1))
                    self.k_list_a = []
                    self.a_list = []
                    self.psi_delta_list = []

        return psi

    def reset(self):
        self.integrated = 0


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
