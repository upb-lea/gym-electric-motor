from .continuous_controller import ContinuousController
from .torque_to_current_conversion import TorqueToCurrentConversion
from .plot_external_data import plot
from gym.spaces import Box
import numpy as np


class CascadedFieldOrientedController:
    """
        This controller is used for torque or speed control of synchronous motors. The controller consists of a field
        oriented controller for current control, an efficiency-optimized torque controller and an optional speed
        controller. The current control is equivalent to the current control of the FieldOrientedController. The torque
        controller is based on the maximum torque per current (MTPC) control strategy in the voltage control range and
        the maximum torque per flux (MTPF) control strategy with an additional modulation controller in the flux
        weakening range. The speed controller is designed as a PI-controller by default.
    """

    def __init__(self,  environment, stages, _controllers, ref_states, external_ref_plots=(), plot_torque=True,
                 plot_modulation=False, update_interval=1000, torque_control='interpolate', **controller_kwargs):
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self.backward_transformation = (lambda quantities, eps: t32(q(quantities, eps)))
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
        if self.current_control:
            self.ref_d_idx = np.where(ref_states == 'i_sd')[0][0]
            self.ref_idx = np.where(ref_states != 'i_sd')[0][0]

        self.omega_control = 'omega' in ref_states and type(environment)
        self.has_cont_action_space = type(self.action_space) is Box

        self.limit = environment.physical_system.limits
        self.nominal_values = environment.physical_system.nominal_state

        self.mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_p = self.mp.get('psi_p', 0)
        self.dead_time = 1.5 if environment.physical_system.converter._dead_time else 0.5
        self.decoupling = controller_kwargs.get('decoupling', True)

        self.ref_state_idx = [self.i_sq_idx, self.i_sd_idx]

        # Initialize torque controller
        if self.torque_control:
            self.ref_state_idx.append(self.torque_idx)
            self.torque_controller = TorqueToCurrentConversion(environment, plot_torque, plot_modulation,
                                                               update_interval, torque_control)
        if self.omega_control:
            self.ref_state_idx.append(self.omega_idx)

        self.ref_idx = 0

        # Initialize continuous controller stages
        if self.has_cont_action_space:
            assert len(stages[0]) == 2, 'Number of stages not correct'
            self.d_controller = _controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[0][0], _controllers, **controller_kwargs)
            self.q_controller = _controllers[stages[0][1]['controller_type']][1].make(
                environment, stages[0][1], _controllers, **controller_kwargs)
            [self.u_sq_0, self.u_sd_0] = [0, 0]

            if self.omega_control:
                self.overlaid_controller = [_controllers[stages[1][i]['controller_type']][1].make(
                    environment, stages[1][i], _controllers, cascaded=True, **controller_kwargs) for i in range(0, len(stages[1]))]
                self.overlaid_type = [_controllers[stages[1][i]['controller_type']][1] == ContinuousController for i in
                                      range(0, len(stages[1]))]

        # Initialize discrete controller stages
        else:

            if self.omega_control:
                assert len(stages) == 4, 'Number of stages not correct'
                self.overlaid_controller = [_controllers[stages[3][i]['controller_type']][1].make(
                    environment, stages[3][i], cascaded=True, **controller_kwargs) for i in range(len(stages[3]))]
                self.overlaid_type = [_controllers[stages[3][i]['controller_type']][1] == ContinuousController for i in
                                      range(len(stages[3]))]
            else:
                assert len(stages) == 3, 'Number of stages not correct'
            self.abc_controller = [_controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[i][0], _controllers, **controller_kwargs) for i in range(3)]
            self.i_abc_idx = [environment.state_names.index(state) for state in ['i_a', 'i_b', 'i_c']]

        self.ref = np.zeros(len(self.ref_state_idx))    # Define array for reference values

        # Set up the plots
        plot_ref = np.append(np.array([environment.state_names[i] for i in self.ref_state_idx]), ref_states)
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(plot_ref)

    def control(self, state, reference):
        """
            Main method that is called by the user to calculate the manipulated variable.

            Args:
                state: state of the gem environment
                reference: reference for the controlled states

            Returns:
                action: action for the gem environment
        """

        self.ref[-1] = reference[self.ref_idx]  # Set the reference

        epsilon_d = state[self.eps_idx] * self.limit[self.eps_idx] + self.dead_time * self.tau * state[self.omega_idx] * \
                    self.limit[self.omega_idx] * self.mp['p']   # Calculate delta epsilon

        # Iterate through high-level controller
        if self.omega_control:
            for i in range(len(self.overlaid_controller) + 1, 1, -1):
                # Calculate reference
                self.ref[i] = self.overlaid_controller[i-2].control(state[self.ref_state_idx[i + 1]], self.ref[i + 1])

                # Check limits and integrate
                if (0.85 * self.state_space.low[self.ref_state_idx[i]] <= self.ref[i] <= 0.85 *
                        self.state_space.high[self.ref_state_idx[i]]) and self.overlaid_type[i - 2]:
                    self.overlaid_controller[i - 2].integrate(state[self.ref_state_idx[i + 1]], self.ref[i + 1])
                else:
                    self.ref[i] = np.clip(self.ref[i], self.nominal_values[self.ref_state_idx[i]] / self.limit[
                        self.ref_state_idx[i]] * self.state_space.low[self.ref_state_idx[i]],
                                          self.nominal_values[self.ref_state_idx[i]] / self.limit[
                                              self.ref_state_idx[i]] * self.state_space.high[self.ref_state_idx[i]])

        # Calculate reference values for i_d and i_q
        if self.torque_control:
            torque = self.ref[2] * self.limit[self.torque_idx]
            self.ref[0], self.ref[1] = self.torque_controller.control(state, torque)

        # Calculate action for continuous action space
        if self.has_cont_action_space:

            # Decouple the two current components
            if self.decoupling:
                self.u_sd_0 = -state[self.omega_idx] * self.mp['p'] * self.mp['l_q'] * state[self.i_sq_idx]\
                              * self.limit[self.i_sq_idx] / self.limit[self.u_sd_idx] * self.limit[self.omega_idx]
                self.u_sq_0 = state[self.omega_idx] * self.mp['p'] * (
                        state[self.i_sd_idx] * self.mp['l_d'] * self.limit[self.u_sd_idx] + self.psi_p) / self.limit[
                         self.u_sq_idx] * self.limit[self.omega_idx]

            # Calculate action for u_sd
            if self.torque_control:
                u_sd = self.d_controller.control(state[self.i_sd_idx], self.ref[1]) + self.u_sd_0
            else:
                u_sd = self.d_controller.control(state[self.i_sd_idx], reference[self.ref_d_idx]) + self.u_sd_0

            # Calculate action for u_sq
            u_sq = self.q_controller.control(state[self.i_sq_idx], self.ref[0]) + self.u_sq_0

            # Shifting the reference potential
            action_temp = self.backward_transformation((u_sd, u_sq), epsilon_d)
            action_temp = action_temp - 0.5 * (max(action_temp) + min(action_temp))

            # Check limit and integrate
            action = np.clip(action_temp, self.action_space.low[0], self.action_space.high[0])
            if (action == action_temp).all():
                if self.torque_control:
                    self.d_controller.integrate(state[self.i_sd_idx], self.ref[1])
                else:
                    self.d_controller.integrate(state[self.i_sd_idx], reference[self.ref_d_idx])
                self.q_controller.integrate(state[self.i_sq_idx], self.ref[0])

        # Calculate action for discrete action space
        else:
            ref = self.ref[1] if self.torque_control else reference[self.ref_d_idx]
            ref_abc = self.backward_transformation((ref, self.ref[0]), epsilon_d)
            action = 0
            for i in range(3):
                action += (2 ** (2 - i)) * self.abc_controller[i].control(state[self.i_abc_idx[i]], ref_abc[i])

        # Plot overlaid reference values
        plot(external_reference_plots=self.external_ref_plots, state_names=self.state_names, external_data=self.get_plot_data(),
             visualization=True)

        return action

    def get_plot_data(self):
        # Getting the external data that should be plotted
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1], external=[])

    def reset(self):
        # Reset the Controllers
        if self.omega_control:
            for overlaid_controller in self.overlaid_controller:
                overlaid_controller.reset()
        if self.has_cont_action_space:
            self.d_controller.reset()
            self.q_controller.reset()

        else:
            for abc_controller in self.abc_controller:
                abc_controller.reset()

        if self.torque_control:
            self.torque_controller.reset()
