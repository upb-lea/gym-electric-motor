from ..core import ElectricMotorEnvironment
from .observer import Observer


class StateObserver(Observer):
    env = None

    def __init__(self, env: ElectricMotorEnvironment):
        self.env = env

    def observe(self, state: str) -> any:
        state_value = self.pull(state)
        self.push(state_value)

    def pull(self, state: str) -> any:
        pass

    def push(self, value):
        print(f"PUSH | {value} \n")
