from .continuous_controller import ContinuousController
from .induction_motor_torque_to_current_conversion import InductionMotorTorqueToCurrentConversion
from .flux_observer import FluxObserver
from .plot_external_data import plot
from gym.spaces import Box
import numpy as np


class InductionMotorCascadedFieldOrientedController:
    """
        This controller is used for torque or speed control of induction motors. The controller consists of a field
        oriented controller for current control, an efficiency-optimized torque controller and an optional speed
        controller. The current control is equivalent to the current control of the FieldOrientedControllerRotorFluxObserver.
        The TorqueToCurrentConversionRotorFluxObserver is used for torque control and a PI-Controller by default is used
        for speed control.
    """

    def __init__(self, environment, stages, _controllers, ref_states, external_ref_plots=(), external_plot=(),
                 **controller_kwargs):

        self.env = environment
        self.action_space = environment.action_space
        self.has_cont_action_space = type(self.action_space) is Box
        self.state_space = environment.physical_system.state_space
        self.state_names = environment.state_names

        self.stages = stages
        self.flux_observer = FluxObserver(self.env)
        self.i_sd_idx = self.env.state_names.index('i_sd')
        self.i_sq_idx = self.env.state_names.index('i_sq')
        self.u_s_abc_idx = [self.env.state_names.index(state) for state in ['u_sa', 'u_sb', 'u_sc']]
        self.omega_idx = self.env.state_names.index('omega')
        self.torque_idx = self.env.state_names.index('torque')

        mp = self.env.physical_system.electrical_motor.motor_parameter
        self.p = mp['p']
        self.l_m = mp['l_m']
        self.l_sigma_s = mp['l_sigs']
        self.l_r = self.l_m + mp['l_sigr']
        self.l_s = self.l_m + mp['l_sigs']
        self.r_r = mp['r_r']
        self.r_s = mp['r_s']
        self.tau_r = self.l_r / self.r_r
        self.sigma = (self.l_s * self.l_r - self.l_m ** 2) / (self.l_s * self.l_r)
        self.limits = self.env.physical_system.limits
        self.nominal_values = self.env.physical_system.nominal_state
        self.tau_sigma = self.sigma * self.l_s / (self.r_s + self.r_r * self.l_m ** 2 / self.l_r ** 2)
        self.tau = self.env.physical_system.tau

        self.dq_to_abc_transformation = environment.physical_system.dq_to_abc_space

        self.torque_control = 'torque' in ref_states or 'omega' in ref_states
        self.current_control = 'i_sd' in ref_states
        self.omega_control = 'omega' in ref_states
        self.ref_state_idx = [self.i_sq_idx, self.i_sd_idx]

        if self.current_control:
            self.ref_d_idx = np.where(ref_states == 'i_sd')[0][0]
            self.ref_idx = np.where(ref_states != 'i_sd')[0][0]

        # Initialize torque controller
        if self.torque_control:
            self.ref_state_idx.append(self.torque_idx)
            self.torque_controller = InductionMotorTorqueToCurrentConversion(environment, stages)

        if self.omega_control:
            self.ref_state_idx.append(self.omega_idx)

        self.ref_idx = 0
        self.psi_opt = 0

        # Set up the plots
        self.external_plot = external_plot
        self.external_ref_plots = external_ref_plots
        self.external_ref_plots = external_ref_plots
        plot_ref = np.append(np.array([environment.state_names[i] for i in self.ref_state_idx]), ref_states)
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(plot_ref)

        labels = [
            {'y_label': r"|$\Psi_{r}$|/Vs", 'state_label': r"|$\hat{\Psi}_{r}$|/Vs", 'ref_label': r"|$\Psi_{r}$|$^*$/Vs"},
            {'y_label': r"$\measuredangle\Psi_r$/rad", 'state_label': r"$\measuredangle\hat{\Psi}_r$/rad"}]

        for ext_plot, label in zip(self.external_plot, labels):
            ext_plot.set_label(label)

        # Initialize continuous controllers
        if self.has_cont_action_space:
            assert len(stages[0]) == 2, 'Number of stages not correct'
            self.decoupling = controller_kwargs.get('decoupling', True)
            self.u_sd_0 = self.u_sq_0 = 0
            self.d_controller = _controllers[stages[0][0]['controller_type']][1].make(
                environment, stages[0][0], _controllers, **controller_kwargs)
            self.q_controller = _controllers[stages[0][1]['controller_type']][1].make(
                environment, stages[0][1], _controllers, **controller_kwargs)

            if self.omega_control:
                self.overlaid_controller = [_controllers[stages[1][i]['controller_type']][1].make(environment,
                    stages[1][i], _controllers, cascaded=True, **controller_kwargs) for i in range(0, len(stages[1]))]
                self.overlaid_type = [_controllers[stages[1][i]['controller_type']][1] == ContinuousController for i in
                                      range(0, len(stages[1]))]

        self.ref = np.zeros(len(self.ref_state_idx))    # Define array for reference values

    def control(self, state, reference):
        """
            This main method of the InductionMotorCascadedFieldOrientedController is called by the user. It calculates
             the input voltages u_a,b,c.

            Args:
                state: state of the gem environment
                reference: reference for the controlled states

            Returns:
                action: action for the gem environment
        """

        self.ref[-1] = reference[self.ref_idx]  # Set the reference
        self.psi_abs, self.psi_angle = self.flux_observer.estimate(state * self.limits)     # Estimate the flux

        # Iterate through the overlaid controller stages
        if self.omega_control:
            for i in range(len(self.overlaid_controller) + 1, 1, -1):
                # Calculate reference
                self.ref[i] = self.overlaid_controller[i-2].control(state[self.ref_state_idx[i + 1]], self.ref[i + 1])

                # Check limit and integrate
                if (0.85 * self.state_space.low[self.ref_state_idx[i]] <= self.ref[i] <= 0.85 *
                        self.state_space.high[self.ref_state_idx[i]]) and self.overlaid_type[i - 2]:
                    self.overlaid_controller[i - 2].integrate(state[self.ref_state_idx[i + 1]], self.ref[i + 1])
                else:
                    self.ref[i] = np.clip(self.ref[i], self.nominal_values[self.ref_state_idx[i]] / self.limits[
                        self.ref_state_idx[i]] * self.state_space.low[self.ref_state_idx[i]],
                                          self.nominal_values[self.ref_state_idx[i]] / self.limits[
                                              self.ref_state_idx[i]] * self.state_space.high[self.ref_state_idx[i]])

        # Calculate reference values for i_d and i_q
        if self.torque_control:
            torque = self.ref[2] * self.limits[self.torque_idx]
            self.ref[0], self.ref[1], self.psi_opt = self.torque_controller.control(state, torque, self.psi_abs)

        if self.has_cont_action_space:
            state = state * self.limits     # Denormalize the state
            omega_me = state[self.omega_idx]
            i_sd = state[self.i_sd_idx]
            i_sq = state[self.i_sq_idx]
            omega_s = omega_me + self.r_r * self.l_m / self.l_r * i_sq / max(np.abs(self.psi_abs), 1e-4) * np.sign(self.psi_abs)

            # Calculate delate u_sd, u_sq
            u_sd_delta = self.d_controller.control(state[self.i_sd_idx],
                                                   self.ref[1] * self.limits[self.i_sd_idx])
            u_sq_delta = self.q_controller.control(state[self.i_sq_idx],
                                                   self.ref[0] * self.limits[self.i_sq_idx])

            # Decouple the two current components
            if self.decoupling:
                self.u_sd_0 = -omega_s * self.sigma * self.l_s * i_sq - self.l_m * self.r_r / (self.l_r ** 2) * self.psi_abs
                self.u_sq_0 = omega_s * self.sigma * self.l_s * i_sd + omega_me * self.l_m / self.l_r * self.psi_abs

            u_sd = self.u_sd_0 + u_sd_delta
            u_sq = self.u_sq_0 + u_sq_delta

            # Transform action in abc coordinates and normalize action
            u_s_abc = self.dq_to_abc_transformation((u_sd, u_sq), self.psi_angle)
            u_s_abc /= self.limits[self.u_s_abc_idx]

            # Limit the action and integrate
            action = np.clip(u_s_abc, self.action_space.low, self.action_space.high)
            if (action == u_s_abc).all():
                self.d_controller.integrate(state[self.i_sd_idx],
                                            self.ref[1] * self.limits[self.i_sd_idx])
                self.q_controller.integrate(state[self.i_sq_idx],
                                            self.ref[0] * self.limits[self.i_sq_idx])

        # Plot the external data
        plot(external_reference_plots=self.external_ref_plots, state_names=self.state_names,
             external_plot=self.external_plot, external_data=self.get_plot_data())

        return action

    def get_plot_data(self):
        # Getting the external data that should be plotted
        return dict(ref_state=self.ref_state_idx[:-1], ref_value=self.ref[:-1],
                    external=[[self.psi_abs, self.psi_opt],
                              [self.psi_angle]])

    def reset(self):
        # Reset the Controllers and the observer
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

        self.flux_observer.reset()
