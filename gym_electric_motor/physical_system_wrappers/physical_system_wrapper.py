import gym_electric_motor as gem


class PhysicalSystemWrapper(gem.core.PhysicalSystem, gem.core.RandomComponent):
    """A PhysicalSystemWrapper is a wrapper around the PhysicalSystem of a gem-environment.

    It may be used to modify its states and actions. In contrast to gym-wrappers which are put around the whole
    environment, modified states by the PhysicalSystemWrappers can be referenced, rewarded and visualized by the
    other components of the environment.
    """

    @property
    def k(self):
        return self._physical_system.k

    @property
    def action_space(self):
        """The processed action space.

        If it is unset, the action space of the inner physical system is returned."""
        if self._action_space is None:
            return self._physical_system.action_space
        return self._action_space

    @action_space.setter
    def action_space(self, space):
        self._action_space = space

    @property
    def state_space(self):
        """The processed state space.

        If it is unset, the state space of the inner physical system is returned."""
        if self._state_space is None:
            return self._physical_system.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space):
        self._state_space = space

    @property
    def physical_system(self):
        """The next inner physical_system or the next inner physical_system_wrapper."""
        return self._physical_system

    @property
    def limits(self):
        """The processed physical system limits.

        If it is unset, the inner limits are returned."""
        if self._limits is None:
            return self._physical_system.limits
        return self._limits

    @property
    def unwrapped(self):
        """The innermost physical system within all Physical System Wrappers."""
        return self.physical_system.unwrapped

    @property
    def nominal_state(self):
        """The processed physical system nominal state.

        If it is unset, the inner nominal state is returned."""
        if self._nominal_state is None:
            return self._physical_system.nominal_state
        return self._nominal_state

    @property
    def state_names(self):
        """The processed physical system state names.

        If it is unset, the state names of the inner physical system are returned."""
        if self._state_names is None:
            return self._physical_system.state_names
        return self._state_names

    def __init__(self, physical_system=None):
        """
        Args:
            physical_system(PhysicalSystem): (Optional) The inner physical system can be passed to directly call the
                set_physical_system-method after the initialization.
        """
        gem.core.PhysicalSystem.__init__(self, None, None, (), None)
        gem.core.RandomComponent.__init__(self)
        self._physical_system = physical_system
        self._limits = None
        self._nominal_state = None
        if physical_system is not None:
            self.set_physical_system(physical_system)

    def set_physical_system(self, physical_system):
        """Sets the inner physical system of this PhysicalSystemWrapper.

        Args:
            physical_system(PhysicalSystem): The inner physical system or Physical System Wrapper.
        """
        self._physical_system = physical_system
        self._action_space = physical_system.action_space
        self._state_names = physical_system.state_names
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        self._tau = physical_system.tau
        return self

    def seed(self, seed=None):
        # docstring of super class RandomComponent
        if isinstance(self._physical_system, gem.core.RandomComponent):
            self._physical_system.seed(seed)

    def __getattr__(self, name):
        return getattr(self._physical_system, name)

    def simulate(self, action):
        # Docstring of super class PhysicalSystem
        return self._physical_system.simulate(action)

    def reset(self, **kwargs):
        # Docstring of super class PhysicalSystem
        self.next_generator()
        return self._physical_system.reset(**kwargs)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.physical_system)

    def __repr__(self):
        return str(self)
