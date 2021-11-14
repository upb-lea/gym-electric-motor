import gym_electric_motor as gem


class StateActionProcessor(gem.core.PhysicalSystem):

    @property
    def action_space(self):
        if self._action_space is None:
            return self._physical_system.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def state_space(self):
        if self._state_space is None:
            return self._physical_system.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space):
        self._state_space = space

    @property
    def physical_system(self):
        return self._physical_system

    @property
    def limits(self):
        if self._limits is None:
            return self._physical_system.limits
        return self._limits

    @property
    def unwrapped(self):
        return self.physical_system.unwrapped

    @property
    def nominal_state(self):
        if self._nominal_state is None:
            return self._physical_system.nominal_state
        return self._nominal_state

    @property
    def state_names(self):
        if self._state_names is None:
            return self._physical_system.state_names
        return self._state_names

    def __init__(self):
        super().__init__(None, None, (), None)
        self._physical_system = None
        self._limits = None
        self._nominal_state = None

    def set_physical_system(self, physical_system):
        self._physical_system = physical_system
        self._action_space = physical_system.action_space
        self._state_names = physical_system.state_names
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        self._tau = physical_system.tau
        return self

    def __getattr__(self, name):
        return getattr(self._physical_system, name)

    def simulate(self, action):
        return self._physical_system.simulate(action)

    def reset(self, **kwargs):
        return self._physical_system.reset(**kwargs)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)
