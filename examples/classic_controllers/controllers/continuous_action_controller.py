from gymnasium.spaces import Box
from gym_electric_motor.physical_systems import DcMotorSystem, DcExternallyExcitedMotor
from .plot_external_data import plot
import numpy as np


class ContinuousActionController:
    """
    This class performs a current-control for all continuous DC motor systems. By default, a PI controller is used
    for current control. An EMF compensation is applied. For the externally excited dc motor, the excitation current
    is also controlled.
    """

    def __init__(
        self,
        environment,
        stages,
        _controllers,
        ref_states,
        external_ref_plots=(),
        **controller_kwargs,
    ):
        assert type(environment.action_space) is Box and isinstance(
            environment.physical_system, DcMotorSystem
        ), "No suitable action space for Continuous Action Controller"
        self.action_space = environment.action_space
        self.state_names = environment.state_names

        self.ref_idx = np.where(ref_states != "i_e")[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.u_idx = environment.physical_system.VOLTAGES_IDX[-1]
        self.limit = environment.physical_system.limits[environment.state_filter]
        self.nominal_values = environment.physical_system.nominal_state[
            environment.state_filter
        ]
        self.omega_idx = self.state_names.index("omega")

        self.action = np.zeros(self.action_space.shape[0])
        self.control_e = isinstance(
            environment.physical_system.electrical_motor, DcExternallyExcitedMotor
        )

        mp = environment.physical_system.electrical_motor.motor_parameter
        self.psi_e = mp.get("psi_e", None)
        self.l_e = mp.get("l_e_prime", None)

        self.action_limit_low = (
            self.action_space.low[0]
            * self.nominal_values[self.u_idx]
            / self.limit[self.u_idx]
        )
        self.action_limit_high = (
            self.action_space.high[0]
            * self.nominal_values[self.u_idx]
            / self.limit[self.u_idx]
        )

        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

        # Initialize Controller
        if self.control_e:  # Check, if a controller for i_e is needed
            assert len(stages) == 2, "Controller design is incomplete"
            assert "i_e" in ref_states, "No reference for i_e"
            self.ref_e_idx = np.where(ref_states == "i_e")[0][0]
            self.controller_e = _controllers[stages[1][0]["controller_type"]][1].make(
                environment, stages[1][0], _controllers, **controller_kwargs
            )
            self.controller = _controllers[stages[0][0]["controller_type"]][1].make(
                environment, stages[0][0], _controllers, **controller_kwargs
            )
            u_e_idx = self.state_names.index("u_e")
            self.action_e_limit_low = (
                self.action_space.low[1]
                * self.nominal_values[u_e_idx]
                / self.limit[u_e_idx]
            )
            self.action_e_limit_high = (
                self.action_space.high[1]
                * self.nominal_values[u_e_idx]
                / self.limit[u_e_idx]
            )
        else:
            if "i_e" not in ref_states:
                assert len(ref_states) <= 1, "Too many referenced states"
            self.controller = _controllers[stages[0]["controller_type"]][1].make(
                environment, stages[0], _controllers, **controller_kwargs
            )

    def control(self, state, reference):
        """
        Main method that is called by the user to calculate the manipulated variable.

        Args:
            state: state of the gem environment
            reference: reference for the controlled states

        Returns:
            action: action for the gem environment
        """
        self.action[0] = self.controller.control(
            state[self.ref_state_idx], reference[self.ref_idx]
        ) + self.feedforward(state)  # Calculate action

        # Limit the action and integrate the I-Controller
        if self.action_limit_low <= self.action[0] <= self.action_limit_high:
            self.controller.integrate(
                state[self.ref_state_idx], reference[self.ref_idx]
            )
        else:
            self.action[0] = np.clip(
                self.action[0], self.action_limit_low, self.action_limit_high
            )

        # Check, if an i_e Controller is used
        if self.control_e:
            # Calculate action
            self.action[1] = self.controller_e.control(
                state[self.i_idx], reference[self.ref_e_idx]
            )
            # Limit the action and integrate the I-Controller
            if self.action_e_limit_low <= self.action[1] <= self.action_e_limit_high:
                self.controller_e.integrate(
                    state[self.i_idx], reference[self.ref_e_idx]
                )
            else:
                self.action[1] = np.clip(
                    self.action[1], self.action_e_limit_low, self.action_e_limit_high
                )

        plot(self.external_ref_plots, self.state_names)  # Plot the external data

        return self.action

    @staticmethod
    def get_plot_data():
        # Getting the external data that should be plotted
        return dict(ref_state=[], ref_value=[], external=[])

    def reset(self):
        # Reset the Controllers
        self.controller.reset()
        if self.control_e:
            self.controller_e.reset()

    def feedforward(self, state):
        # EMF compensation
        psi_e = (
            self.psi_e or self.l_e * state[self.i_idx] * self.nominal_values[self.i_idx]
        )
        return (
            state[self.omega_idx] * self.nominal_values[self.omega_idx] * psi_e
        ) / self.nominal_values[self.u_idx]
