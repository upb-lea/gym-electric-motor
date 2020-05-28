from ..core import ElectricMotorVisualization
import numpy as np

class ConsolePrinter(ElectricMotorVisualization):

    _limits = None

    def __init__(self, print_freq = None, **kwargs):
        """
        Args:
            print_freq(String):
                String indicating whether and at which frequency the console will be printed.
                Options:
                    None: No Printing
                    'step': Printing at each step of an episode
                    'episode': Printing after an episode has terminated
        """
        
        self._print_freq = print_freq
        self._num_steps = 0
        self._cum_reward = 0
        self._violation = False
        np.set_printoptions(formatter={'float': '{:0>7.3f}'.format})
            
    def reset(self, reference_trajectories=None, *_, **__):
        if self._print_freq == 'episode' and self._violation == False:
            print(f'External Reset. Number of steps: {self._num_steps: 08d} Cumulated Reward: {self._cum_reward:7.3f}')
        else:
            self._violation = False
        self._num_steps = 0
        self._cum_reward = 0


    def set_modules(self, physical_system, *_, **__):
        self._limits = physical_system.limits

    def step(self, state, reference, reward, done, *_, **__):
        self._num_steps += 1
        self._cum_reward += reward
        if self._print_freq == 'step':
            print(f'State {state * self._limits} Reference {reference * self._limits} Reward {reward:7.3f} Step {self._num_steps:08d} Cumulated Reward {self._cum_reward:7.3f}' , end='\r')
        elif self._print_freq == 'episode' and done:
            print(f'Constraint Violation! Number of steps: {self._num_steps: 08d} Cumulated Reward: {self._cum_reward:7.3f}')
            self._violation = True
            
