import numpy as np

import gem_controllers as gc
import gym_electric_motor as gem

from .gem_controller import GemController


class GymElectricMotorAdapter(GemController):
    """The GymElectricMotorAdapter wraps a GemController to map the inputs and outputs to the environment."""

    @property
    def input_stage(self):
        """Input stage of the controller"""
        return self._input_stage

    @input_stage.setter
    def input_stage(self, value):
        self._input_stage = value

    @property
    def output_stage(self):
        """Output stage of the controller"""
        return self._output_stage

    @output_stage.setter
    def output_stage(self, value):
        self._output_stage = value

    @property
    def controller(self):
        """Wrapped GemController"""
        return self._controller

    @controller.setter
    def controller(self, value):
        self._controller = value

    @property
    def block_diagram(self):
        return self._block_diagram

    def __init__(
        self,
        _env: (gem.core.ElectricMotorEnvironment, None) = None,
        env_id: (str, None) = None,
        controller: (GemController, None) = None,
    ):
        """
        Args:
            _env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            controller(GemController): The GemController that should be wrapped.
        """

        super().__init__()
        self._input_stage = None
        self._output_stage = None
        assert isinstance(controller, GemController)
        self._controller = controller
        self._input_stage = gc.stages.InputStage()
        action_type = gc.utils.get_action_type(env_id)
        if action_type == "Finite":
            self._output_stage = gc.stages.DiscOutputStage()
        else:
            self._output_stage = gc.stages.ContOutputStage()
        self._block_diagram = None
        self._reference_plotter = gc.ReferencePlotter()

    def control(self, state, reference):
        """
        Function to calculate the action of the controller for the environment.

        Args:
            state(np.array): Array of the state of the environment.
            reference(np.array): Array of the references of the referenced states.

        Returns:
            action
        """

        # Copy state and reference to be independent from further calculations
        state_, reference_ = np.copy(state), np.copy(reference)

        # Denormalize the state and reference
        denormalized_ref = self._input_stage(state_, reference_)

        # Iterate through the controller stages to calculate the input voltages
        voltage_set_point = self._controller.control(state_, denormalized_ref)

        # Transform and normalize the input voltages
        action = self._output_stage(state_, voltage_set_point)
        if self.should_plot:
            self._reference_plotter.update_plots(self._controller.references)
        return action

    def tune(self, env, env_id, tune_controller=True, **kwargs):
        """
        Function to set the parameters of the controller stages.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be tuned for.
            env_id(str): ID of the ElectricMotorEnvironment.
            tune_controller(bool): Flag, if the controller should be tuned.
        """

        self._input_stage.tune(env, env_id)
        self._output_stage.tune(env, env_id)
        if tune_controller:
            self._controller.tune(env, env_id, **kwargs)

        self._reference_plotter.tune(
            env,
            self._controller.referenced_states,
            plot_references=True,
            maximum_reference=self._controller.maximum_reference,
        )

    def build_block_diagram(self, env_id, save_block_diagram_as):
        self._block_diagram = gc.build_block_diagram(self, env_id, save_block_diagram_as)

    def reset(self):
        """Reset all stages of the controller."""
        self._input_stage.reset()
        self._controller.reset()
        self._output_stage.reset()
