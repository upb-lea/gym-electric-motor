import gym_electric_motor as gem


class RandomReferenceGenerator(gem.ReferenceGenerator, gem.RandomComponent):

    def __init__(self):
        gem.ReferenceGenerator.__init__(self)
        gem.RandomComponent.__init__(self)

    def reset(self, initial_state=None, initial_reference=None):
        self.next_generator()

    def get_reference(self, state, *_, **__):
        # Docs from superclass
        raise NotImplementedError

    def get_reference_observation(self, state, *_, **__):
        # Docs from superclass
        raise NotImplementedError
