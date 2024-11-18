import numpy as np


class RandomComponent:
    """Base class for all random components in the environment.

    Every component (e.g. ReferenceGenerator or ElectricMotor) that has any kind of random behavior has to derive from
    the RandomComponent class directly or indirectly via its own base class.

    During the seeding of the environment, the random components 'seed' function is called and a sub-seed is passed
    for every random component. From this sub-seed, further new seeds and reference generators
    are generated at the beginning of each episode. This ensures a reproducible random behavior for each new episode
    no matter how long the previous episodes have been. Therefore, each random environment component has to call the
    'next_generator()' method during reset.

    Everytime a random number is used, it has to be drawn from the ``random_generator``.

    Example:

        >>> import gym_electric_motor as gem
        >>> import numpy as np
        >>>
        >>>
        >>> class MyRandomReferenceGenerator(gem.ReferenceGenerator, gem.RandomComponent):
        >>>
        >>>     def __init__(self):
        >>>         gem.ReferenceGenerator.__init__(self)
        >>>         gem.RandomComponent.__init__(self)
        >>>         self.my_random_var = np.array([0.0]) # Only for exemplary purpose
        >>>         # Further custom initialization
        >>>         # ...
        >>>
        >>>     def reset(self, initial_state=None, initial_reference=None):
        >>>         self.next_generator()
        >>>         # Reset further components states
        >>>
        >>>     def get_reference(self, state, *_, **__):
        >>>         # Draw from the classes random_generator
        >>>         self.my_random_var = self.random_generator.normal(size=1)
        >>>         # Do something with the random variable e.g.:
        >>>         return np.ones_like(state) * self.my_random_var
        >>>
        >>>     def get_reference_observation(self, state, *_, **__):
        >>>         # Do sth.
        >>>         return self.my_random_var
        >>>

    """

    @property
    def random_generator(self):
        """The random generator that has to be used to draw the random numbers."""
        return self._random_generator

    @property
    def seed_sequence(self):
        """The base seed sequence that generates the sub generators and sub seeds at every environment reset."""
        return self._seed_sequence

    def __init__(self):
        self._seed_sequence = np.random.SeedSequence()
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])

    def seed(self, seed=None):
        """The function to set the seed.

        This function is called by within the global seed call of the environment. The environment passes the sub-seed
        to this component that is generated based on the source-seed of the env.

        Args:
            seed((np.random.SeedSequence, None)): Seed sequence to derive new seeds and reference generators at every
                episode start. Default: None (a new SeedSequence is generated).

        Returns:
            List(int): A list containing all seeds within this RandomComponent. In general, this list has length 1.
            If the RandomComponent holds further RandomComponent instances, the list has to contain also these
            entropies. The entropy of this instance has to be placed always at first place.
        """
        if seed is None:
            seed = np.random.SeedSequence()
        self._seed_sequence = seed
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])
        return [self._seed_sequence.entropy]

    def next_generator(self):
        """Sets a new reference generator for a new episode."""
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])
