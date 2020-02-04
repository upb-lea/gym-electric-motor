from ..core import ElectricMotorVisualization


class ConsolePrinter(ElectricMotorVisualization):

    _limits = None

    def __init__(self, **kwargs):
        pass

    def reset(self, reference_trajectories=None, *_, **__):
        pass

    def set_physical_system(self, physical_system):
        self._limits = physical_system.limits

    def step(self, state, reference, reward, *_, **__):
        print(
            f'State {state * self._limits} \n'
            f'Reference {reference * self._limits}\n'
            f'Reward {reward}\n'
            f'\n'
        )
