from ..core import ElectricMotorVisualization
import numpy as np


class ConsolePrinter(ElectricMotorVisualization):
    """Prints current training values of the environment on the console.

    These include environment state and reference values as well as the number of training steps, the rewards
    and the cumulative reward. It also shows why an episode has terminated (external, constraint violation)
    """

    def __init__(self, verbose=0, update_freq=1):
        """
        Args:
            verbose(Int):
                Integer indicating whether and at which frequency the console will be printed.
                Options:
                    0: No Printing
                    1: Printing after an episode has terminated
                    2: Printing at each step of an episode
            update_freq(Int):
                Unsigned Integer setting the update frequency if verbose is 2.
                It's value n means that each nth step the corresponding output will be printed
        """
        super().__init__()
        self._limits = None
        self._state = np.nan
        self._reference = np.nan
        self._action = np.nan
        self._reward = np.nan
        self._k = 0
        self._print_freq = verbose
        self._update_freq = update_freq
        self._num_steps = 0
        self._cum_reward = 0
        self._done = False
        self._episode = 0
        np.set_printoptions(formatter={'float': '{:9.3f}'.format})
        self._reset = False

    def on_reset_begin(self):
        """Gets called on environment reset. Handles internal value reset and External reset printing"""
        self._reset = True

    def on_reset_end(self, state, reference):
        self._reference = reference
        self._state = state

    def set_env(self, env):
        """Gets the limits of the current physical system for accurate printing"""
        self._limits = env.physical_system.limits

    def on_step_begin(self, k, action):
        self._k = k
        self._action = action

    def on_step_end(self, k, state, reference, reward, done):
        """Gets called at each step of the environment.
        Handles per step printing as well es constraint violation printing"""
        self._k = k
        self._state = state
        self._reference = reference
        self._reward = reward
        self._cum_reward += reward
        self._done = done

    def render(self):
        if self._print_freq > 0:
            if self._reset:
                print(
                    f'\nEpisode {self._episode} ',
                    f'Constraint Violation! ' if self._done else 'External Reset. ',
                    f'Number of steps: {self._k: 8d} ',
                    f'Cumulative Reward: {self._cum_reward:7.3f}\n')
                self._cum_reward = 0
                self._reset = False
                self._episode += 1

            if self._print_freq == 2 and (self._k % self._update_freq) == 0:
                print(f'Episode {self._episode} '
                      f'Step {self._k: 8d} '
                      f'State {self._state * self._limits} '
                      f'Reference {self._reference * self._limits} '
                      f'Reward {self._reward:7.3f} '
                      f'Cumulative Reward {self._cum_reward:7.3f}',
                      end='\r'
                      )
