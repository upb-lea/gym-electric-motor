from ..core import ElectricMotorVisualization
import numpy as np

class ConsolePrinter(ElectricMotorVisualization):
    """
    Prints current training values of the environment on the console. These include environment state and reference values as well as
    the number of training steps, the rewards and the cumulated reward. Also shows why an episode has terminated (external, constraint violation)
    """


    def __init__(self, verbose=0, update_freq=1, **kwargs):
        """
        Args:
            verbose(Int):
                Integer indicating whether and at which frequency the console will be printed.
                Options:
                    0: No Printing
                    1: Printing after an episode has terminated
                    2: Printing at each step of an episode
            update_freq(Int):
                Unsigned Integer setting the update frequency if verbose is 2. It's value n means that each nth step the corresponding output will be printed
                    
        """
        self._limits = None
        self._print_freq = verbose
        self._update_freq = update_freq
        self._num_steps = 0
        self._cum_reward = 0
        self._violation = False
        self._episode = 0
        np.set_printoptions(formatter={'float': '{:0=9.3f}'.format})
            
    def reset(self, *_, **__):
        """Gets called on environment reset. Handles internal value reset and External reset printing"""
        if self._print_freq:
            if not self._violation:
                print(
                      f'\nEpisode {self._episode} '
                      f'External Reset. ' 
                      f'Number of steps: {self._num_steps: 08d} '
                      f'Cumulated Reward: {self._cum_reward:7.3f}')
            
            self._violation = False
            self._num_steps = 0
            self._cum_reward = 0
        self._episode += 1

   
            


    def set_modules(self, physical_system, *_, **__):
        """Gets the limits of the current physical system for accurate printing"""
        self._limits = physical_system.limits

    def step(self, state, reference, reward, done, *_, **__):
        """Gets called at each step of the environment. Handles per step printing as well es constraint violation printing"""
        if self._print_freq:
            if not self._num_steps:
                print('\n')
            self._num_steps += 1
            self._cum_reward += reward
            if self._print_freq == 2 and not self._num_steps%self._update_freq:
                print(  f'Episode {self._episode} '
                        f'State {state * self._limits} '
                        f'Reference {reference * self._limits} '
                        f'Reward {reward:7.3f} '
                        f'Step {self._num_steps:08d} '
                        f'Cumulated Reward {self._cum_reward:7.3f}' 
                        , end='\r'
                        )
            if done:
                print(f'\nEpisode {self._episode} Constraint Violation! Number of steps: {self._num_steps: 08d} Cumulated Reward: {self._cum_reward:7.3f}')
                self._violation = True
            
