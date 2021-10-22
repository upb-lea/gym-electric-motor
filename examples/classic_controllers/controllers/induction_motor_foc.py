from .flux_observer import FluxObserver
from .plot_external_data import plot
from gym.spaces import Box
import numpy as np


class InductionMotorFieldOrientedController:
    """
        This class controls the currents of induction motors using a field oriented controller. The control is performed
        in the rotating dq-stator-coordinates. For this purpose, the two current components are optionally decoupled and
        two independent current controllers are used. The rotor flux required for this is estimated based on a current
        model.
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
        self.i_sd_ref_idx = np.where(ref_states == 'i_sd')[0][0]
        self.i_sq_ref_idx = np.where(ref_states == 'i_sq')[0][0]
        self.omega_idx = self.env.state_names.index('omega')

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
        self.tau_sigma = self.sigma * self.l_s / (self.r_s + self.r_r * self.l_m**2 / self.l_r**2)
        self.tau = self.env.physical_system.tau

        self.dq_to_abc_transformation = environment.physical_system.dq_to_abc_space

        # Set up the plots
        self.external_plot = external_plot
        self.external_ref_plots = external_ref_plots

        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

        labels = [{'y_label': r"|$\Psi_{r}$|/Vs", 'state_label': r"|$\hat{\Psi}_{r}$|/Vs"},
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

    def control(self, state, reference):
        """
            This main method of the InductionMotorFieldOrientedController is called by the user. It calculates the input
            voltages u_a,b,c.

            Args:
                state: state of the gem environment
                reference: reference for the controlled states

            Returns:
                action: action for the gem environment
        """

        state = state * self.limits     # Denormalize the state
        self.psi_abs, self.psi_angle = self.flux_observer.estimate(state)  # Estimate the flux

        omega_me = state[self.omega_idx]
        i_sd = state[self.i_sd_idx]
        i_sq = state[self.i_sq_idx]

        omega_s = omega_me + self.r_r * self.l_m / self.l_r * i_sq / max(np.abs(self.psi_abs), 1e-4) * np.sign(self.psi_abs)

        # Calculate delate u_sd, u_sq
        u_sd_delta = self.d_controller.control(state[self.i_sd_idx], reference[self.i_sd_ref_idx] * self.limits[self.i_sd_idx])
        u_sq_delta = self.q_controller.control(state[self.i_sq_idx], reference[self.i_sq_ref_idx] * self.limits[self.i_sq_idx])

        # Decouple the two current components
        if self.decoupling:
            self.u_sd_0 = -omega_s * self.sigma * self.l_s * i_sq - self.l_m * self.r_r / (self.l_r ** 2) * self.psi_abs
            self.u_sq_0 = omega_s * self.sigma * self.l_s * i_sd + omega_me * self.l_m / self.l_r * self.psi_abs

        u_sd = self.u_sd_0 + u_sd_delta
        u_sq = self.u_sq_0 + u_sq_delta

        # Transform action in abc coordinates and normalize action
        u_s_abc = self.dq_to_abc_transformation((u_sd, u_sq), self.psi_angle)
        u_s_abc /= self.limits[self.u_s_abc_idx]

        # Limit action and integrate
        action = np.clip(u_s_abc, self.action_space.low, self.action_space.high)
        if (action == u_s_abc).all():
            self.d_controller.integrate(state[self.i_sd_idx], reference[self.i_sd_ref_idx] * self.limits[self.i_sd_idx])
            self.q_controller.integrate(state[self.i_sq_idx], reference[self.i_sq_ref_idx] * self.limits[self.i_sq_idx])

        # Plot the external data
        plot(external_plot=self.external_plot, external_data=self.get_plot_data())

        return action

    def get_plot_data(self):
        # Getting the external data that should be plotted
        return dict(ref_state=[], ref_value=[], external=[[self.psi_abs], [self.psi_angle]])

    def reset(self):
        # Reset the Controllers and the observer
        self.flux_observer.reset()
        self.d_controller.reset()
        self.q_controller.reset()
