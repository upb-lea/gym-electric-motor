import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from gym_electric_motor.physical_systems import DcExternallyExcitedMotor

from .continuous_controller import ContinuousController
from .plot_external_data import plot


class CascadedController:
    """
    This class is used for cascaded torque and speed control of all dc motor environments. Each stage can contain
    continuous or discrete controllers. For the externally excited dc motor an additional controller is used for
    the excitation current. The calculated reference values of the intermediate stages can be inserted into the
    plots.
    """

    def __init__(
        self,
        environment,
        stages,
        _controllers,
        visualization,
        ref_states,
        external_ref_plots=(),
        **controller_kwargs,
    ):
        self.env = environment
        self.visualization = visualization
        self.action_space = environment.action_space
        self.state_space = environment.unwrapped.physical_system.state_space
        self.state_names = environment.unwrapped.state_names

        self.i_e_idx = environment.unwrapped.physical_system.CURRENTS_IDX[-1]
        self.i_a_idx = environment.unwrapped.physical_system.CURRENTS_IDX[0]
        self.u_idx = environment.unwrapped.physical_system.VOLTAGES_IDX[-1]
        self.omega_idx = environment.unwrapped.state_names.index("omega")
        self.torque_idx = environment.unwrapped.state_names.index("torque")
        self.ref_idx = np.where(ref_states != "i_e")[0][0]
        self.ref_state_idx = [
            self.i_a_idx,
            environment.unwrapped.state_names.index(ref_states[self.ref_idx]),
        ]

        self.limit = environment.unwrapped.physical_system.limits[environment.unwrapped.state_filter]
        self.nominal_values = environment.unwrapped.physical_system.nominal_state[environment.unwrapped.state_filter]

        self.control_e = isinstance(environment.unwrapped.physical_system.electrical_motor, DcExternallyExcitedMotor)
        self.control_omega = 0

        mp = environment.unwrapped.physical_system.electrical_motor.motor_parameter
        self.psi_e = mp.get("psie_e", False)
        self.l_e = mp.get("l_e_prime", False)
        self.r_e = mp.get("r_e", None)
        self.r_a = mp.get("r_a", None)

        # Set the action limits
        if type(self.action_space) is Box:
            self.action_limit_low = self.action_space.low[0] * self.nominal_values[self.u_idx] / self.limit[self.u_idx]
            self.action_limit_high = (
                self.action_space.high[0] * self.nominal_values[self.u_idx] / self.limit[self.u_idx]
            )

        # Set the state limits
        self.state_limit_low = self.state_space.low * self.nominal_values / self.limit
        self.state_limit_high = self.state_space.high * self.nominal_values / self.limit

        # Initialize i_e Controller if needed
        if self.control_e:
            assert len(stages) == 2, "Controller design is incomplete"
            self.ref_e_idx = False if "i_e" not in ref_states else np.where(ref_states == "i_e")[0][0]
            self.control_e_idx = 1

            if self.omega_idx in self.ref_state_idx:
                self.ref_state_idx.insert(1, self.torque_idx)
                self.control_omega = 1

            self.ref_state_idx.append(self.i_e_idx)
            self.controller_e = _controllers[stages[1][0]["controller_type"]][1].make(
                environment,
                stages[1][0],
                _controllers,
                control_e=True,
                **controller_kwargs,
            )
            stages = stages[0]
            u_e_idx = self.state_names.index("u_e")

            # Set action limit for u_e
            if type(self.action_space) is Box:
                self.action_e_limit_low = self.action_space.low[1] * self.nominal_values[u_e_idx] / self.limit[u_e_idx]
                self.action_e_limit_high = (
                    self.action_space.high[1] * self.nominal_values[u_e_idx] / self.limit[u_e_idx]
                )

        else:
            self.control_e_idx = 0
            assert len(ref_states) <= 1, "Too many referenced states"

        # Check of the stages are using continuous or discrete controller
        self.stage_type = [_controllers[stage["controller_type"]][1] == ContinuousController for stage in stages]

        # Initialize Controller stages
        self.controller_stages = [
            _controllers[stage["controller_type"]][1].make(
                environment, stage, _controllers, cascaded=stages.index(stage) != 0
            )
            for stage in stages
        ]

        # Set up the plots
        self.external_ref_plots = external_ref_plots
        internal_refs = np.array([environment.unwrapped.state_names[i] for i in self.ref_state_idx])
        ref_states_plotted = np.unique(np.append(ref_states, internal_refs))
        for external_plots in self.external_ref_plots:
            external_plots.set_reference(ref_states_plotted)

        assert type(self.action_space) is Box or not self.stage_type[0], "No suitable inner controller"
        assert (
            type(self.action_space) in [Discrete, MultiDiscrete] or self.stage_type[0]
        ), "No suitable inner controller"

        self.ref = np.zeros(len(self.controller_stages) + self.control_e_idx + self.control_omega)

    def control(self, state, reference):
        """
        Main method that is called by the user to calculate the manipulated variable.

        Args:
            state: state of the gem environment
            reference: reference for the controlled states

        Returns:
            action: action for the gem environment
        """

        # Set the reference
        self.ref[-1 - self.control_e_idx] = reference[self.ref_idx]

        # Iterate through the high-level controller stages
        for i in range(
            len(self.controller_stages) - 1,
            0 + self.control_e_idx - self.control_omega,
            -1,
        ):
            # Set the indices
            ref_idx = i - 1 + self.control_omega
            state_idx = self.ref_state_idx[ref_idx]

            # Calculate reference for lower stage
            self.ref[ref_idx] = self.controller_stages[i].control(state[state_idx], self.ref[ref_idx + 1])

            # Check limits and integrate
            if (
                self.state_limit_low[state_idx] <= self.ref[ref_idx] <= self.state_limit_high[state_idx]
            ) and self.stage_type[i]:
                self.controller_stages[i].integrate(state[self.ref_state_idx[i + self.control_omega]], reference[0])

            elif self.stage_type[i]:
                self.ref[ref_idx] = np.clip(
                    self.ref[ref_idx],
                    self.state_limit_low[state_idx],
                    self.state_limit_high[state_idx],
                )

        # Calculate optimal i_a and i_e for externally excited dc motor
        if self.control_e:
            i_e = np.clip(
                np.power(
                    self.r_a * (self.ref[1] * self.limit[self.torque_idx]) ** 2 / (self.r_e * self.l_e**2),
                    1 / 4,
                ),
                self.state_space.low[self.i_e_idx] * self.limit[self.i_e_idx],
                self.state_space.high[self.i_e_idx] * self.limit[self.i_e_idx],
            )
            i_a = np.clip(
                self.ref[1] * self.limit[self.torque_idx] / (self.l_e * i_e),
                self.state_space.low[self.i_a_idx] * self.limit[self.i_a_idx],
                self.state_space.high[self.i_a_idx] * self.limit[self.i_a_idx],
            )
            self.ref[-1] = i_e / self.limit[self.i_e_idx]
            self.ref[0] = i_a / self.limit[self.i_a_idx]

        # Calculate action for u_a
        action = self.controller_stages[0].control(state[self.ref_state_idx[0]], self.ref[0])

        # Check if stage is continuous
        if self.stage_type[0]:
            action += self.feedforward(state)  # EMF compensation

            # Check limits and integrate
            if self.action_limit_low <= action <= self.action_limit_high:
                self.controller_stages[0].integrate(state[self.ref_state_idx[0]], self.ref[0])
                action = [action]
            else:
                action = np.clip([action], self.action_limit_low, self.action_limit_high)

        # Calculate action for u_e if needed
        if self.control_e:
            if self.ref_e_idx:
                self.ref[-1] = reference[self.ref_e_idx]
            action_u_e = self.controller_e.control(state[self.i_e_idx], self.ref[-1])

            # Check limits and integrate
            if self.stage_type[0]:
                action = np.append(action, action_u_e)
                if self.action_e_limit_low <= action[1] <= self.action_e_limit_high:
                    self.controller_e.integrate(state[self.i_e_idx], self.ref[-1])
                action = np.clip(action, self.action_e_limit_low, self.action_e_limit_high)
            else:
                action = np.array([action, action_u_e], dtype="object")

        if self.env.render_mode != None:
            # Plot the external references
            plot(
                external_reference_plots=self.external_ref_plots,
                state_names=self.state_names,
                visualization=self.visualization,
                external_data=self.get_plot_data(),
            )

        return action

    def feedforward(self, state):
        # EMF compensation
        psi_e = max(
            self.psi_e or self.l_e * state[self.i_e_idx] * self.nominal_values[self.i_e_idx],
            1e-6,
        )
        return (state[self.omega_idx] * self.nominal_values[self.omega_idx] * psi_e) / self.nominal_values[self.u_idx]

    def get_plot_data(self):
        # Getting the external data that should be plotted
        return dict(ref_state=self.ref_state_idx, ref_value=self.ref, external=[])

    def reset(self):
        # Reset the Controllers
        for controller in self.controller_stages:
            controller.reset()
        if self.control_e:
            self.controller_e.reset()
