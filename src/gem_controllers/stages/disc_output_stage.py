import gymnasium
import numpy as np

import gem_controllers as gc
from gym_electric_motor.physical_systems import converters as cv

from .. import parameter_reader as reader
from ..utils import non_parameterized
from .stage import Stage


class DiscOutputStage(Stage):
    """This class maps the discrete input voltages, calculated by the controller, to the scalar inputs of the used
    converter.
    """

    @property
    def output_stage(self):
        """Output stage of the controller"""
        return self._output_stage

    @output_stage.setter
    def output_stage(self, value):
        assert value in [self.to_b6_discrete, self.to_multi_discrete, self.to_discrete]
        self._output_stage = value

    def __init__(self):
        super().__init__()
        self.high_level = 0.0
        self.low_level = 0.0
        self.high_action = 0
        self.low_action = 0
        self.idle_action = 0
        self._output_stage = non_parameterized

    def __call__(self, state, reference):
        """
        Maps the input voltages to the scalar inputs of the converter.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference voltages.

        Returns:
            action(int): scalar action of the environment
        """
        return self._output_stage(self.to_action(state, reference))

    @staticmethod
    def to_discrete(multi_discrete_action):
        """
        Transform multi discrete action to a discrete action.

        Args:
            multi_discrete_action(np.array): Array of multi discrete actions

        Returns:
            int: discrete action
        """
        return multi_discrete_action[0]

    @staticmethod
    def to_b6_discrete(multi_discrete_action):
        """Returns the multi discrete action for a B6 brigde converter."""
        raise NotImplementedError

    @staticmethod
    def to_multi_discrete(multi_discrete_action):
        """
        Returns the multi discrete action.
        """
        return multi_discrete_action

    def to_action(self, _state, reference):
        """
        Map the voltages to a voltage level.

        Args:
             _state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference voltages.

        Returns:
            action(np.ndarray): volatge vector
        """
        conditions = [reference <= self.low_level, reference >= self.high_level]
        return np.select(conditions, [self.low_action, self.high_action], default=self.idle_action)

    def tune(self, env, env_id, **__):
        """
        Set the values for the low, idle and high action.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        action_type, _, motor_type = gc.utils.split_env_id(env_id)
        voltages = reader.get_output_voltages(motor_type, action_type)
        voltage_indices = [env.state_names.index(voltage) for voltage in voltages]
        voltage_limits = env.limits[voltage_indices]

        voltage_range = (
            env.observation_space[0].low[voltage_indices] * voltage_limits,
            env.observation_space[0].high[voltage_indices] * voltage_limits,
        )

        self.low_level = -0.33 * (voltage_range[1] - voltage_range[0])
        self.high_level = 0.33 * (voltage_range[1] - voltage_range[0])

        if type(env.action_space) == gymnasium.spaces.MultiDiscrete:
            self.output_stage = DiscOutputStage.to_multi_discrete
            self.low_action = []
            self.idle_action = []
            self.high_action = []
            for n in env.action_space.nvec:
                low_action, idle_action, high_action = self._get_actions(n)
                self.low_action.append(low_action)
                self.idle_action.append(idle_action)
                self.high_action.append(high_action)

        elif (
            type(env.action_space) == gymnasium.spaces.Discrete
            and type(env.physical_system.converter) != cv.FiniteB6BridgeConverter
        ):
            self.output_stage = DiscOutputStage.to_discrete
            self.low_action, self.idle_action, self.high_action = self._get_actions(env.action_space.n)
        elif type(env.physical_system.converter) == cv.FiniteB6BridgeConverter:
            self.output_stage = DiscOutputStage.to_b6_discrete
        else:
            raise Exception(f"No discrete output stage available for action space {env.action_space}.")

    @staticmethod
    def _get_actions(n):
        high_action = 1
        if n == 2:  # OneQuadrantConverter
            low_action = 0
        else:  # Two and FourQuadrantConverter
            low_action = 2
        idle_action = 0
        return low_action, idle_action, high_action
