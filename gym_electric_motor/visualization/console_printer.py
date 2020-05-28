from ..core import ElectricMotorVisualization


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
        
        #Assert a class variable to an integer depending on the 
        self._print_freq = print_freq
        self._num_steps = 0
        self._cum_reward = 0
        self._violation = False
            
        #pass

    def reset(self, reference_trajectories=None, *_, **__):
        if self._print_freq == 'episode' and self._violation == False:
            print('Termination because of external reset')
            print('Total number of steps in this episode: ', self._num_steps)
            print('Cumulated Reward of this episode: ', self._cum_reward)
            self._num_steps = 0
            self._cum_reward = 0
        else:
            self._violation = False

    def set_modules(self, physical_system, *_, **__):
        self._limits = physical_system.limits

    def step(self, state, reference, reward, done, *_, **__):
        self._num_steps += 1
        self._cum_reward += reward
        if self._print_freq == 'step':
            print(
                f'State {state * self._limits} \n'
                f'Reference {reference * self._limits}\n'
                f'Reward {reward}\n'
                f'\n'
            )
        elif self._print_freq == 'episode' and done:
            print('Termination because of constraint violation')
            print('Total number of steps in this episode: ', self._num_steps)
            print('Cumulated Reward of this episode: ', self._cum_reward)
            self._violation = True
            self._num_steps = 0
            self._cum_reward = 0
            
