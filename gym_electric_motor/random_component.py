import numpy as np


class RandomComponent:

    @property
    def random_generator(self):
        return self._random_generator

    @property
    def seed_sequence(self):
        return self._seed_sequence

    def __init__(self):
        self._seed_sequence = np.random.SeedSequence()
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])

    def seed(self, seed=None):
        self._seed_sequence = seed
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])
        return [self._seed_sequence.entropy]

    def next_generator(self):
        self._random_generator = np.random.default_rng(self._seed_sequence.spawn(1)[0])
