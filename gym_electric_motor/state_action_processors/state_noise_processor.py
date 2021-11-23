from gym_electric_motor.state_action_processors import StateActionProcessor


class StateNoiseProcessor(StateActionProcessor):

    @property
    def random_kwargs(self):
        return self._random_kwargs

    @random_kwargs.setter
    def random_kwargs(self, value):
        self._random_kwargs = dict(value)

    def __init__(self, states, random_dist='normal', random_kwargs=(), random_length=1000, physical_system=None):
        self._random_kwargs = dict(random_kwargs)
        self._random_length = int(random_length)
        self._random_pointer = 0
        self._noise = None
        self._states = states
        self._random_dist = random_dist
        self._state_indices = []
        super().__init__(physical_system)
        assert hasattr(self._random_generator, random_dist), \
            f'The numpy random number generator has no distribution {random_dist}.'\
            'Check https://numpy.org/doc/stable/reference/random/generator.html#distributions for distributions.'

    def set_physical_system(self, physical_system):
        super().set_physical_system(physical_system)
        self._state_indices = [physical_system.state_positions[state_name] for state_name in self._states]
        return self

    def reset(self):
        state = super().reset()
        self._new_noise()
        return self._add_noise(state)

    def simulate(self, action):
        if self._random_pointer >= self._random_length:
            self._new_noise()
        return self._add_noise(self._physical_system.simulate(action))

    def _add_noise(self, state):
        state[self._state_indices] = state[self._state_indices] + self._noise[self._random_pointer]
        self._random_pointer += 1
        return state

    def _new_noise(self):
        self._random_pointer = 0
        fct = getattr(self._random_generator, self._random_dist)
        self._noise = fct(size=(self._random_length, len(self._state)), **self._random_kwargs)
