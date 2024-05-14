import numpy as np

import gem_controllers as gc
import gym_electric_motor as gem

from .stages import BaseController
from .torque_controller import TorqueController


class PISpeedController(gc.GemController):
    """This class forms the PI speed controller, for any motor."""

    @property
    def speed_control_stage(self) -> gc.stages.BaseController:
        """Base controller of the speed controller stage"""
        return self._speed_control_stage

    @speed_control_stage.setter
    def speed_control_stage(self, value: gc.stages.BaseController):
        self._speed_control_stage = value

    @property
    def torque_controller(self) -> TorqueController:
        """Subordinated torque controller stage"""
        return self._torque_controller

    @torque_controller.setter
    def torque_controller(self, value: TorqueController):
        self._torque_controller = value

    @property
    def torque_reference(self) -> np.ndarray:
        """Reference values of the torque controller stage"""
        return self._torque_reference

    @property
    def anti_windup_stage(self):
        """Anti windup stage of the speed controller"""
        return self._anti_windup_stage

    @property
    def clipping_stage(self):
        """Clipping stage of the speed controller"""
        return self._clipping_stage

    @property
    def references(self):
        refs = self._torque_controller.references
        refs.update(dict(torque=self._torque_reference[0]))
        return refs

    @property
    def referenced_states(self):
        return np.append(self._torque_controller.referenced_states, "torque")

    @property
    def maximum_reference(self):
        return self._torque_controller.maximum_reference

    def __init__(
        self,
        _env: (gem.core.ElectricMotorEnvironment, None) = None,
        env_id: (str, None) = None,
        torque_controller: (TorqueController, None) = None,
        base_speed_controller: str = "PI",
    ):
        """
        Initilizes a PI speed control stage.

        Args:
            _env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            torque_controller(TorqueController): The underlying torque control stage
            base_speed_controller(str): Selection which base controller should be used for the speed control stage.
        """

        super().__init__()
        self._speed_control_stage = gc.stages.base_controllers.get(base_speed_controller)("SC")
        self._torque_controller = torque_controller
        if torque_controller is None:
            self._torque_controller = TorqueController()
        self._torque_reference = np.array([])
        self._anti_windup_stage = gc.stages.AntiWindup("SC")
        self._clipping_stage = gc.stages.clipping_stages.AbsoluteClippingStage("SC")

    def tune(self, env, env_id, tune_torque_controller=True, a=4, **kwargs):
        """
        Tune the components of the current control stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            tune_torque_controller(bool): Flag, if the underlying torque control stage should be tuned.
            a(float): Design parameter of the symmetric optimum for the base controllers
        """

        if tune_torque_controller:
            self._torque_controller.tune(env, env_id, a=a, **kwargs)
        self._anti_windup_stage.tune(env, env_id)
        self._clipping_stage.tune(env, env_id)
        t_n = min(self._torque_controller.t_n)  # Get the time constant of the torque control stage
        self._speed_control_stage.tune(env, env_id, t_n=t_n, a=a)

    def speed_control(self, state, reference):
        """
        Calculate the torque reference.

        Args:
            state(np.array): actual state of the environment
            reference(np.array): actual speed references

        Returns:
            torque_reference(np.array)
        """

        # Calculate the torque reference by the base controller
        torque_reference = self._speed_control_stage(state, reference)

        # Clipping the torque reference and integrating the I-Controller
        self._torque_reference = self._clipping_stage(state, torque_reference)
        if hasattr(self._speed_control_stage, "integrator"):
            delta = self._anti_windup_stage.__call__(state, reference, self._clipping_stage.clipping_difference)
            self._speed_control_stage.integrator += delta

        return self._torque_reference

    def control(self, state, reference):
        """
        Claculate the reference values for the input voltages.

        Args:
            state(np.array): actual state of the environment
            reference(np.array): speed references

        Returns:
            np.ndarray: voltage reference
        """

        # Calculate the torque reference
        reference = self.speed_control(state, reference)

        # Calculate the references of the underlying stages
        reference = self._torque_controller.control(state, reference)

        return reference

    def reset(self):
        """Reset all components of the speed control stage and the underlying control stages"""
        self._torque_controller.reset()
        self._speed_control_stage.reset()
