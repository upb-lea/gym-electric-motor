from .plot_external_data import plot
from gym_electric_motor.physical_systems import SynchronousMotorSystem
from gym.spaces import Box
import numpy as np


class FieldOrientedController:
    """
        This class controls the currents of synchronous motors. In the case of continuous manipulated variables, the
        control is performed in the rotating dq-coordinates. For this purpose, the two current components are optionally
        decoupled and two independent current controllers are used.
        In the case of discrete manipulated variables, control takes place in stator-fixed coordinates. The reference
        values are converted into these coordinates so that a on-off controller calculates the corresponding
        manipulated variable for each current component.
    """

    def __init__(self, environment, stages, _controllers, ref_states, external_ref_plots=[], **controller_kwargs):
        assert isinstance(environment.physical_system, SynchronousMotorSystem), 'No suitable Environment for FOC Controller'

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
        self.psi_p = self.mp.get('psi_p', 0)
        self.dead_time = 1.5 if environment.physical_system.converter._dead_time else 0.5
        self.has_cont_action_space = type(self.action_space) is Box
        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

        if self.has_cont_action_space:
            assert len(stages[0]) == 2, 'Number of stages not correct'
            self.decoupling = controller_kwargs.get('decoupling', True)
            [self.u_sq_0, self.u_sd_0] = [0, 0]

            self.d_controller = _controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[0][0], _controllers, **controller_kwargs)
            self.q_controller = _controllers[stages[0][1]['controller_type']][1].make(
                environment, stages[0][1], _controllers, **controller_kwargs)

        else:
            assert len(stages) == 3, 'Number of stages not correct'
            self.abc_controller = [_controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[i][0], _controllers, **controller_kwargs) for i in range(3)]
            self.i_abc_idx = [environment.state_names.index(state) for state in ['i_a', 'i_b', 'i_c']]

    def control(self, state, reference):
        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + self.dead_time * self.tau * \
                    state[self.omega_idx] * self.limit[self.omega_idx] * self.mp['p']
        if self.has_cont_action_space:
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
            if (action == action_temp).all():
                self.d_controller.integrate(state[self.d_idx], reference[self.ref_d_idx])
                self.q_controller.integrate(state[self.q_idx], reference[self.ref_q_idx])

        else:
            ref_abc = self.backward_transformation((reference[self.ref_q_idx], reference[self.ref_d_idx]), epsilon_d)
            action = 0
            for i in range(3):
                action += (2 ** (2 - i)) * self.abc_controller[i].control(state[self.i_abc_idx[i]], ref_abc[i])

        plot(self.external_ref_plots, self.state_names, external_data=self.get_plot_data())
        return action

    def get_plot_data(self):
        return dict(ref_state=[], ref_value=[], external=[])

    def reset(self):
        if self.has_cont_action_space:
            self.d_controller.reset()
            self.q_controller.reset()
