import numpy as np

import gem_controllers as gc
import gym_electric_motor as gem


class TorqueController(gc.GemController):
    """This class forms the torque controller, for any motor."""

    @property
    def torque_to_current_stage(self) -> gc.stages.OperationPointSelection:
        """Operation point selection stage"""
        return self._operation_point_selection

    @torque_to_current_stage.setter
    def torque_to_current_stage(self, value: gc.stages.OperationPointSelection):
        self._operation_point_selection = value

    @property
    def current_controller(self) -> gc.CurrentController:
        """Subordinated current controller stage"""
        return self._current_controller

    @current_controller.setter
    def current_controller(self, value: gc.CurrentController):
        self._current_controller = value

    @property
    def current_reference(self) -> np.ndarray:
        """Reference values of the current controller stage"""
        return self._current_reference

    @property
    def clipping_stage(self) -> gc.stages.clipping_stages.ClippingStage:
        """Clipping stage of the torque controller stage"""
        return self._clipping_stage

    @clipping_stage.setter
    def clipping_stage(self, value: gc.stages.clipping_stages.ClippingStage):
        self._clipping_stage = value

    @property
    def t_n(self):
        """Time constant of the current controller stage"""
        return self._current_controller.t_n

    @property
    def references(self):
        refs = self._current_controller.references
        refs.update(dict(zip(self._referenced_currents, self._current_reference)))
        return refs

    @property
    def referenced_states(self):
        return np.append(self._current_controller.referenced_states, self._referenced_currents)

    @property
    def maximum_reference(self):
        return self._maximum_reference

    def __init__(
        self,
        env: (gem.core.ElectricMotorEnvironment, None) = None,
        env_id: (str, None) = None,
        current_controller: (gc.CurrentController, None) = None,
        torque_to_current_stage: (gc.stages.OperationPointSelection, None) = None,
        clipping_stage: (gc.stages.clipping_stages.ClippingStage, None) = None,
    ):
        """
        Initilizes a torque control stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            current_controller(gc.CurrentController): The underlying current control stage.
            torque_to_current_stage(gc.stages.OperationPointSelection): The operation point selection class of the
             torque contol stage.
            clipping_stage(gc.stages.clipping_stages.ClippingStage): Clipping stage of the torque control stage.
        """

        super().__init__()

        self._operation_point_selection = torque_to_current_stage
        if env_id is not None and torque_to_current_stage is None:
            self._operation_point_selection = gc.stages.torque_to_current_function[gc.utils.get_motor_type(env_id)]()
        self._current_controller = current_controller
        if env_id is not None and current_controller is None:
            self._current_controller = gc.PICurrentController(env, env_id)
        if env_id is not None and clipping_stage is None:
            if gc.utils.get_motor_type(env_id) in gc.parameter_reader.dc_motors:
                self._clipping_stage = gc.stages.clipping_stages.AbsoluteClippingStage("TC")
            elif gc.utils.get_motor_type(env_id) == "EESM":
                self._clipping_stage = gc.stages.clipping_stages.CombinedClippingStage("TC")
            else:  # motor in ac_motors
                self._clipping_stage = gc.stages.clipping_stages.SquaredClippingStage("TC")
        self._current_reference = np.array([])
        self._referenced_currents = np.array([])
        self._maximum_reference = dict()

    def tune(self, env, env_id, current_safety_margin=0.2, tune_current_controller=True, **kwargs):
        """
        Tune the components of the current control stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            current_safety_margin(float): Percentage indicating the maximum value for the current reference.
            tune_current_controller(bool): Flag, if the underlying current control stage should be tuned.
        """

        if tune_current_controller:
            self._current_controller.tune(env, env_id, **kwargs)
        self._clipping_stage.tune(env, env_id, margin=current_safety_margin)
        self._operation_point_selection.tune(env, env_id, current_safety_margin)
        self._referenced_currents = gc.parameter_reader.currents[gc.utils.get_motor_type(env_id)]
        for current, action_range_low, action_range_high in zip(
            self._referenced_currents, self._clipping_stage.action_range[0], self._clipping_stage.action_range[1]
        ):
            if current in ["i", "i_a", "i_e"]:
                self._maximum_reference[current] = [action_range_low, action_range_high]

    def torque_control(self, state, reference):
        """
        Calculate the current refrences.

        Args:
            state(np.array): actual state of the environment
            reference(np.array): actual torque references

        Returns:
            current references(np.array)
        """

        self._current_reference = self._operation_point_selection(state, reference)
        return self._current_reference

    def control(self, state, reference):
        """
        Claculate the reference values for the input voltages.

        Args:
            state(np.array): state of the environment
            reference(np.array): torque references

        Returns:
            np.ndarray: voltage reference
        """

        # Calculate the current references
        self._current_reference = self.torque_control(state, reference)

        # Clipping the current references
        self._current_reference = self._clipping_stage(state, self._current_reference)

        # Calculate the voltage reference
        reference = self._current_controller.current_control(state, self._current_reference)

        return reference

    def reset(self):
        """Reset all components of the torque control stage and the underlying stages"""
        self._current_controller.reset()
        self._operation_point_selection.reset()
        self._clipping_stage.reset()
