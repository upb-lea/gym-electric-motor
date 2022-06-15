from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class StateNoiseProcessor(PhysicalSystemWrapper):
    """The StateNoiseProcessor puts additional noise onto the systems state.

    The random distribution of the noise can be selected by those available in the numpy random generator:
            `<https://numpy.org/doc/stable/reference/random/generator.html#distributions>`_

    Example:
        .. code-block:: python
            import gym_electric_motor as gem
            state_noise_processor = gem.physical_system_wrappers.StateNoiseProcessor(
                states=['omega', 'torque'],
                random_dist='laplace'
                random_kwargs=dict(loc=0.0, scale=0.1)
            )
            env = gem.make('Cont-SC-PermExDc-v0', physical_system_wrappers=(state_noise_processor,))


    """

    @property
    def random_kwargs(self):
        """Returns the random keyword arguments that are passed through to the random generator function."""
        return self._random_kwargs

    @random_kwargs.setter
    def random_kwargs(self, value):
        self._random_kwargs = dict(value)

    def __init__(self, states, random_dist='normal', random_kwargs=(), random_length=1000, physical_system=None):
        """
        Args:
             states(Iterable[string] / 'all'): Names of the states onto which the noise shall be added.
                Shortcut 'all': Add it to all available states.
            random_dist(string): Name of the random distribution behind the numpy random generator that shall be used.
            random_kwargs(dict): Keyword arguments that are passed through to the selected random distribution.
                For available entries have a look at the page linked above. Only the argument 'size' cannot be passed.
            random_length(int): Number of random samples that are drawn at once. Drawing from the random distribution
                at every cycle would be too slow. So, multiple samples are drawn at once for the next steps.
            physical_system(PhysicalSystem): (Optional) If available the inner physical system can already be passed to
                the processor here. Otherwise the processor will be initialized during environment creation.
        """
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
        # Docstring from super class
        super().set_physical_system(physical_system)
        self._state_indices = [physical_system.state_positions[state_name] for state_name in self._states]
        return self

    def reset(self):
        # Docstring from super class
        state = super().reset()
        self._new_noise()
        return self._add_noise(state)

    def simulate(self, action):
        # Docstring from super class
        if self._random_pointer >= self._random_length:
            self._new_noise()
        return self._add_noise(self._physical_system.simulate(action))

    def _add_noise(self, state):
        """Adds noise to the current state.
        Args:
            state(numpy.ndarray[float]): The systems state without noise.
        Returns:
            numpy.ndarray[float]): The state with additional noise.
        """
        state[self._state_indices] = state[self._state_indices] + self._noise[self._random_pointer]
        self._random_pointer += 1
        return state

    def _new_noise(self):
        """Samples new noise from the random distribution for the next steps."""
        self._random_pointer = 0
        fct = getattr(self._random_generator, self._random_dist)
        self._noise = fct(size=(self._random_length, len(self._state_indices)), **self._random_kwargs)
