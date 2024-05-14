import abc

from ..core import ElectricMotorEnvironment


class Observer:
    env = None

    def __init__(self, env: ElectricMotorEnvironment):
        self.env = env

    def observe(self, *args, **kwargs) -> any:
        result = self.pull(*args, **kwargs)
        self.push(result)
        return result

    @abc.abstractmethod
    def pull(self, *args, **kwargs) -> any:
        pass

    @abc.abstractmethod
    def push(self, result):
        pass


class StateObserver(Observer):
    state_names = None

    def __init__(self, env: ElectricMotorEnvironment):
        super().__init__(env)
        self.fuse_state_and_next_reference()

    def fuse_state_and_next_reference(self):
        state_names = self.env.unwrapped.state_names
        reference_names = self.env.unwrapped.reference_names
        for ref_name in reference_names:
            state_names.append(ref_name + "_ref")
        self.state_names = state_names

    def pull(self, state: str) -> any:
        # Integrate reference into states

        state_names = self.state_names

        if state in state_names:
            state_value = self.env.current_state[state_names.index(state)]
            if state_value is not None:
                return (state, state_value)
            else:
                raise ValueError(f"State '{state}' is None")
        else:
            raise ValueError(f"State '{state}' not found in state_names, allowed states are {state_names}")

    def push(self, value):
        pass
