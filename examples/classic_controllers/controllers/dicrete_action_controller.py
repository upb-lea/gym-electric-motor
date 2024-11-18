from .plot_external_data import plot
from gymnasium.spaces import Discrete, MultiDiscrete
from gym_electric_motor.physical_systems import DcMotorSystem, DcExternallyExcitedMotor
import numpy as np


class DiscreteActionController:
    """
    This class is used for current control of all DC motor systems with discrete actions. By default, a three-point
    controller is used. For the externally excited dc motor, the excitation current is also controlled.
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
        assert type(environment.action_space) in [
            Discrete,
            MultiDiscrete,
        ] and isinstance(
            environment.physical_system, DcMotorSystem
        ), "No suitable action space for Discrete Action Controller"

        self.ref_idx = np.where(ref_states != "i_e")[0][0]
        self.ref_state_idx = environment.state_names.index(ref_states[self.ref_idx])
        self.i_idx = environment.physical_system.CURRENTS_IDX[-1]
        self.control_e = isinstance(
            environment.physical_system.electrical_motor, DcExternallyExcitedMotor
        )
        self.state_names = environment.state_names

        self.external_ref_plots = external_ref_plots
        for ext_ref_plot in self.external_ref_plots:
            ext_ref_plot.set_reference(ref_states)

        # Initialize Controller
        if self.control_e:  # Check, if a controller for i_e is needed
            assert len(stages) == 2, "Controller design is incomplete"
            assert "i_e" in ref_states, "No reference for i_e"
            self.ref_e_idx = np.where(ref_states == "i_e")[0][0]
            self.controller_e = _controllers[stages[1][0]["controller_type"]][1].make(
                environment,
                stages[1][0],
                _controllers,
                control_e=True,
                **controller_kwargs,
            )
            self.controller = _controllers[stages[0][0]["controller_type"]][1].make(
                environment, stages[0][0], **controller_kwargs
            )
        else:
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
        plot(self.external_ref_plots, self.state_names)  # Plot external data

        # Check if i_e controller is used
        if self.control_e:
            return [
                self.controller.control(
                    state[self.ref_state_idx], reference[self.ref_idx]
                ),
                self.controller_e.control(state[self.i_idx], reference[self.ref_e_idx]),
            ]
        else:
            return self.controller.control(
                state[self.ref_state_idx], reference[self.ref_idx]
            )

    @staticmethod
    def get_plot_data():
        # Get external plot data
        return dict(ref_state=[], ref_value=[], external=[])

    def reset(self):
        # Reset the Controllers
        self.controller.reset()
        if self.control_e:
            self.control_e.reset()
