import gym_electric_motor as gem


class PhysicalSystemTestWrapper(gem.physical_system_wrappers.PhysicalSystemWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action = None
        self.state = None

    def simulate(self, action):
        self.action = action
        self.state = self.physical_system.simulate(action)
        return self.state

    def reset(self):
        self.state = self.physical_system.reset()
        return self.state
