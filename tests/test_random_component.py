import pytest
import gym_electric_motor as gem
import numpy as np


class TestRandomComponent:

    @pytest.fixture
    def random_component(self):
        return gem.RandomComponent()

    def test_seed(self, random_component):
        """Test, if a predefined SeedSequence is set when it is passed."""
        seed_sequence = np.random.SeedSequence()
        random_component.seed(seed_sequence)
        assert seed_sequence == random_component.seed_sequence

    def test_default_seed(self, random_component):
        """Test, if a random SeedSequence is set when no SeedSequence is passed."""
        random_component.seed()
        assert isinstance(random_component.seed_sequence, np.random.SeedSequence)
        assert isinstance(random_component.random_generator, np.random.Generator)

    def test_reseed(self, random_component):
        """Test if the seed of the RandomComponent differs after two random seedings."""
        random_component.seed()
        initial_seed = random_component.seed_sequence
        random_component.seed()
        assert random_component.seed_sequence != initial_seed

    def test_next_generator(self, random_component):
        """This test checks if the next_generator function sets the RandomComponent to defined states after each
        next_generator() call, no matter how many steps have been taken before."""
        # Seed the random component with a defined seed.
        random_component.seed(np.random.SeedSequence(123))

        # Generate a first episode of 42 random numbers
        rands_first_ep = random_component.random_generator.random(42)
        # Use the next generator
        random_component.next_generator()
        # Generate another episode of 42 steps
        rands_second_ep = random_component.random_generator.random(42)

        # Reseed the environment to the previous state
        random_component.seed(np.random.SeedSequence(123))
        # Test, if the first random numbers of the first episodes are equal
        assert(np.all(rands_first_ep[:30] == random_component.random_generator.random(30))),\
            'The random numbers of the initial and reseeded random component differ.'

        random_component.next_generator()
        # Also the second episode has to be equal. Therefore, the next generator has to be set np matter how many steps
        # have been taken in the first episode.
        assert(np.all(rands_second_ep == random_component.random_generator.random(64)[:42])),\
            'The random numbers of the initial and reseeded random component differ.'
