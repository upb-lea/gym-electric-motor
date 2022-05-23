from collections import deque
import numpy as np

import gym.spaces
from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class DeadTimeProcessor(PhysicalSystemWrapper):
    """The DeadTimeProcessor delays the actions to the physical system for a parameterizable amount of steps.

    Reset Actions:
        When the environment is reset, no valid previous actions are available. Per default, constant reset actions are
        used. Also, custom reset actions parameterized. Therefore, a method can be passed during initialization of a
        DeadTimeProcessor. This method needs to return a list of valid actions of the action space whose length equals
        the number of dead time steps.

        The default reset actions are zeros. The concrete shape is derived by the concrete action spaces:
        ``Discrete, MultiDiscrete, Box``.
    """

    @property
    def dead_time(self):
        """int: The number of delayed steps."""
        return self._steps

    def __init__(self, steps=1, reset_action=None, physical_system=None):
        """Args:
            steps(int): Number of steps to delay the actions.
            reset_action(callable): A callable that returns a list of length steps to initialize the dead-actions
                after a reset. Default: See above in the class description
            physical_system(PhysicalSystem (optional)): The inner physical system of this PhysicalSystemWrapper.
        """
        self._reset_actions = reset_action
        self._steps = int(steps)
        assert self._steps > 0, f'The number of steps has to be greater than 0. A "{steps}" has been passed.'
        self._action_deque = deque(maxlen=steps)
        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        """Sets the inner PhysicalSystem of the DeadTimeProcessor.

        Args:
            physical_system(PhysicalSystem): The physical system to be set.
        """
        super().set_physical_system(physical_system)
        if self._reset_actions is None:
            action_space = physical_system.action_space
            if isinstance(action_space, gym.spaces.Discrete):
                reset_action = 0
            elif isinstance(action_space, gym.spaces.MultiDiscrete):
                reset_action = [np.zeros_like(action_space.nvec)]
            elif isinstance(action_space, gym.spaces.Box):
                reset_action = np.zeros(action_space.shape, dtype=np.float64)
            else:
                raise AssertionError(
                    f'Action Space {action_space} of type {type(action_space)} unsupported.'
                    'Only Discrete / MultiDiscrete and Box allowed for the dead time processor.'
                )
            self._reset_actions = lambda: [reset_action] * self._action_deque.maxlen
        return self

    def reset(self):
        """Resets the processor and the inner physical system for a new episode.

        Returns:
            numpy.ndarray[float]: The initial state of the system.
        """
        state = super().reset()
        self._action_deque.clear()
        self._action_deque.extend(self._reset_actions())
        return state

    def simulate(self, action):
        """Saves the action, applies the dead-time action and simulates the system for one time step.

         Args:
             action(element of the action_space): The next action for the system.

        Returns:
            numpy.ndarray[float]: The next state of the system.
        """
        active_action = self._action_deque.pop()
        self._action_deque.appendleft(action)
        return self._physical_system.simulate(active_action)
