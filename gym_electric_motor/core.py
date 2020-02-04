"""
On the core level the electric motor environment and the interface to its submodules are defined. By using these
interfaces further reference generators, reward functions, visualizations or physical models can be implemented.

Each ElectricMotorEnvironment contains the four following modules:

* PhysicalSystem
    - Specification and simulation of the physical model. Furthermore, specifies limits and nominal values\
    for all of its ``state_variables``.
* ReferenceGenerator
    - Calculation of reference trajectories for one or more states of the physical systems ``state_variables``.
* RewardFunction
    - Calculation of the reward based on the physical systems state and the reference.\
     Furthermore, observation of the physical systems limits.
* ElectricMotorVisualization
    - Visualization of the PhysicalSystems state, reference and reward for the user.
"""
import gym
import numpy as np
from gym.spaces import Box

from .utils import set_state_array
from .utils import instantiate


class ElectricMotorEnvironment(gym.core.Env):
    """
    Description:
        The main class connecting all modules of the gym-electric-motor environments.

    Modules:

        Physical System:
            Containing the physical structure and simulation of the drive system as well as information about the
            technical limits and nominal values. Needs to be a subclass of *PhysicalSystem*

        Reference Generator:
            Generation of the reference for the motor to follow. Needs to be a subclass of *ReferenceGenerator*

        Reward Function:
            Calculation of the reward based on the state of the physical system and the generated reference
            and observation if the motor state is within the limits. Needs to be a subclass of *RewardFunction*.

        Visualization:
            Visualization of the motors states. Needs to be a subclass of *ElectricMotorVisualization*

    State Variables:
        Each environment has got a list of state variables that are defined by the physical system.
        These define the names and order for all further state arrays in the modules. These states are announced to the
        other modules by announcing the physical system to them, which contains the property ``state_names``.

        Example:
            ``['omega', 'torque','i', 'u', 'u_sup']``

    Observation:
        Type: Tuple(State_Space, Reference_Space)
            The observation is always a tuple of the State Space of the Physical System and the Reference Space of the
            Reference Generator. In all current Physical Systems and Reference Generators these Spaces are normalized,
            continuous, multidimensional boxes in [-1, 1] or [0, 1].

    Actions:
        Type: Discrete() / Box()
            The action space of the environments are the action spaces of the physical systems. In all current physical
            systems the action spaces are specified by its PowerElectronicConverter and either a continuous,
            multidimensional box or discrete.

    Reward:
        The reward and the reward range are specified by the RewardFunction. In general the reward is higher the closer
        the motor state follows the reference trajectories.

    Starting State:
        The physical system and the reference generator define the starting state.

    Episode Termination:
        Episode terminations can be initiated by the reference generator, or the reward function.
        A reference generator might terminate an episode, if the reference has ended.
        The reward function can terminate an episode, if a physical limit of the motor has been violated.
    """
    @property
    def physical_system(self):
        """
        Returns:
             PhysicalSystem: The Physical System of the Environment
        """
        return self._physical_system

    @property
    def reference_generator(self):
        """
        Returns:
             ReferenceGenerator: The ReferenceGenerator of the Environment
        """
        return self._reference_generator

    @reference_generator.setter
    def reference_generator(self, reference_generator):
        """
        Setting of a new reference generator for the environment. Afterwards, a reset is required.

        Args:
            reference_generator(ReferenceGenerator): The new reference generator of the environment.
        """
        self._reference_generator = reference_generator
        self._reset_required = True

    @property
    def reward_function(self):
        """
        Returns:
             RewardFunction: The RewardFunction of the environment
        """
        return self._reward_function

    @reward_function.setter
    def reward_function(self, reward_function):
        """
        Setting of a new reward function for the environment. Afterwards, a reset is required.

        Args:
            reward_function(RewardFunction): The new reward function of the environment.
        """
        self._reward_function = reward_function
        self._reset_required = True

    def __init__(self, physical_system, reference_generator, reward_function, visualization=None, state_filter=None,
                 **kwargs):
        """
        Setting and initialization of all environments' modules.

        Args:
            physical_system(PhysicalSystem): The physical system of this environment.
            reference_generator(ReferenceGenerator): The reference generator of this environment.
            reward_function(RewardFunction): The reward function of this environment.
            visualization(ElectricMotorVisualization): The visualization of this environment.
            state_filter(list(str)): Selection of states that are shown in the observation.
            **kwargs: Arguments to be passed to the modules.
        """
        self._physical_system = instantiate(PhysicalSystem, physical_system, **kwargs)
        self._reference_generator = instantiate(ReferenceGenerator, reference_generator, **kwargs)
        self._reward_function = instantiate(RewardFunction, reward_function, **kwargs)
        visualization = visualization or ElectricMotorVisualization()
        self._visualization = instantiate(ElectricMotorVisualization, visualization, **kwargs)

        # Announcement of the Modules among each other
        self._reference_generator.set_modules(self.physical_system)
        self._reward_function.set_modules(self.physical_system, self._reference_generator)
        self._visualization.set_modules(self.physical_system, self._reference_generator, self._reward_function)
        self._reset_required = True

        # Initialization of properties
        self._state = np.zeros(len(self.physical_system.state_names))
        self._reference = np.zeros(len(self.physical_system.state_names))
        self._reward = 0.0

        # Initialization of the state filter and the spaces
        state_filter = state_filter or self._physical_system.state_names
        self._state_filter = []
        for state_var in state_filter:
            self._state_filter.append(np.where(np.array(self._physical_system.state_names) == state_var)[0][0])
        states_low = self._physical_system.state_space.low[self._state_filter]
        states_high = self._physical_system.state_space.high[self._state_filter]
        state_space = Box(states_low, states_high)
        self.observation_space = gym.spaces.Tuple((
            state_space,
            self._reference_generator.reference_space
        ))
        self.action_space = self.physical_system.action_space
        self.reward_range = self._reward_function.reward_range

    def reset(self, *_, **__):
        """
        Reset of the environment and all its modules to an initial state.

        Returns:
             The initial observation consisting of the initial state and initial reference.
        """
        self._reset_required = False
        self._state = self._physical_system.reset()
        self._reference, next_ref, trajectories = self.reference_generator.reset(self._state)
        self._visualization.reset(trajectories)
        self._reward_function.reset(self._state, self._reference)
        self._reward = 0.0
        return self._state[self._state_filter], next_ref

    def render(self, *_, **__):
        """
        Update the visualization of the motor.
        """
        self._visualization.step(self._state, self._reference, self._reward)

    def step(self, action):
        """
        Perform one simulation step of the environment with an action of the action space.

        Args:
            action: Action to play on the environment.

        Returns:
            observation(Tuple(ndarray(float),ndarray(float)): Tuple of the new state and the next reference.
            reward(float): Amount of reward received for the last step.
            done(bool): Flag, indicating if a reset is required before new steps can be taken.
            {}: An empty dictionary for consistency with the OpenAi Gym interface.
        """
        if self._reset_required:
            raise Exception('A reset is required before the environment can perform further steps')
        self._state = self._physical_system.simulate(action)
        self._reference = self.reference_generator.get_reference(self._state)
        self._reward, self._reset_required = self._reward_function.reward(self._state, self._reference, action)

        ref_next = self.reference_generator.get_reference_observation(self._state)
        return (self._state[self._state_filter], ref_next), self._reward, self._reset_required, {}

    def close(self):
        """
        Called when the environment is deleted. Closes all its modules.
        """
        self._reward_function.close()
        self._physical_system.close()
        self._reference_generator.close()
        self._visualization.close()


class ElectricMotorVisualization:
    """
    Base class for all visualizations in the gym-electric-motor toolbox and an empty dummy visualization, if no
    visualization is required.
    """

    def set_modules(self, physical_system, reference_generator, reward_function):
        """
        Args:
            physical_system(PhysicalSystem):
                This function is called from the environment in its initialization.
                Settings that require the knowledge of the physical system (like setting of state variables) have to be
                done in this function.
            reference_generator(ReferenceGenerator):
                This function is called from the environment in its initialization.
                Settings that require the knowledge of the reference generator (like setting of referenced variables)
                have to be done in this function.
            reward_function(RewardFunction):
                This function is called from the environment in its initialization.
                Settings that require the knowledge of the reward function (like setting of reward ranges) have
                to be done in this function.
        """
        pass

    def reset(self, reference_trajectories=None, *_, **__):
        """
        Called when the environment is reset to clear or save plots etc.

        Args:
            reference_trajectories(dict(list/ndarray(float))): If references are known in advance by the reference
            generator, they are passed here to the visualization as a dictionary of the state_name to a
            list of future reference points.
        """
        pass

    def step(self, state, reference, reward, *_, **__):
        """
        Called by the environment every cycle and passes the current, normalized state array,
        the normalized references in the same shape as the state array and the received reward.
        For non-referenced states the value in the reference array can be ignored.

        Example:
            ```state_variables = ['omega', 'torque', 'i', 'u', 'u_sup']```
            ```state = [0.65, 0.32, 0.2, 1.0, 1.0]```
            ```reference = [0.7, 0.0, 0.0, 0.0, 0.0]```
            Only the first reference value is important here, because all others are not referenced.

        Args:
            state(ndarray(float)): The state of the physical system.
            reference(ndarray(float)): The reference array of the reference generator.
            reward(float): The reward from the reward function.
        """
        pass

    def close(self, *_, **__):
        """
        Called when the environment is deleted to close or save figures, logs etc.
        """
        pass


class ReferenceGenerator:
    """
    The abstract base class for reference generators in the gym electric motor environments.

    reference_space:
        Space from the OpenAI Gym Toolbox to define the space of the reference observations.

    The reference generator is called twice per step.

    Call of get_reference():
        Once, to get the reference array. The reference array has got the same shape as the state array and contains
        the current reference values for referenced state variables and a default value (e.g zero) at all other entries.
        This reference array is used to calculate the rewards.

        Example:
            ``reference_array=np.array([0.8, 0.0, 0.0, 0.0])`` \n
            ``state_variables=['omega', 'torque', 'i', 'u', 'u_sup']`` \n
            This would be the corresponding reference array for an environment with the upper state array with an omega
            reference.

    Call of get_reference_observation():
        Another time to get the reference observation. The reference observation needs to be in the defined reference
        space. For example, it can contain future reference values of the next n steps (of the referenced states only).
        It can have in general any shape and content. The returned object of this method is the reference that is
        shown to the agent.

        Example:
            ``reference_observation = np.array([0.8, 0.6, 0.4])`` \n
            This array could model a reference observation for an omega controlled environment that shows the agent not
            only the reference for the next time step t+1 but also t+3 and t+3.

    """

    #: The gym.space the references are in.
    reference_space = None
    _physical_system = None
    _referenced_states = None

    @property
    def referenced_states(self):
        """
        Returns:
            ndarray(bool): Boolean-Array with the length of the state_variables indicating which states are referenced.
        """
        return self._referenced_states

    def set_modules(self, physical_system):
        """
        Announcement of the PhysicalSystem to the ReferenceGenerator.

        In subclasses, store all important information from the physical system to the ReferenceGenerator here.
        The environment announces the physical system to the ReferenceGenerator during its initialization.

        Args:
            physical_system(PhysicalSystem): The physical system of the environment.
        """
        self._physical_system = physical_system

    def get_reference(self, state, *_, **__):
        """
        Returns the reference array of the current time step.

        The reference array needs to be in the same shape as the state variables. For referenced states the reference
        value is passed. For unreferenced states a default value (e.g. Zero) can be set in the reference array.

        Args:
            state(ndarray(float)): Current state array of the environment.

        Returns:
             ndarray(float)): Current reference array.
        """
        raise NotImplementedError

    def get_reference_observation(self, state, *_, **__):
        """
        Returns the reference observation for the next time step. This observation needs to fit in the reference space.

        Args:
            state(ndarray(float)): Current state array of the environment.

        Returns:
            value in reference_space: Observation for the next reference time step.
        """
        raise NotImplementedError

    def reset(self, initial_state=None, initial_reference=None):
        """
        Reset of references for a new episode.

        Args:
            initial_state(ndarray(float)): The initial state array of the environment.
            initial_reference(ndarray(float)): If not None: A desired initial reference array.

        Returns:
            reference_array(ndarray(float)): The current reference array of time step 0.

            reference_observation(value in reference_space): The reference observation for the next time step. \\

            trajectories(dict(list(float)): If available, \
                generated trajectories for the Visualization can be passed here. Otherwise return None. \
        """
        return self.get_reference(initial_state), self.get_reference_observation(initial_state), None

    def close(self):
        """
        Called by the environment, when the environment is deleted to close files, store logs...
        """
        pass


class RewardFunction:
    """
    The abstract base class for reward functions in the gym electric motor environments.

    The reward function is called once per step and returns reward for the current time step.
    Furthermore, the reward function includes the limit observer. If limits have been violated and the episode ended
    a different reward for violating the limits is returned.

    reward_range:
        Tuple(float, float): Defining lowest and highest possible rewards for non limit-violating steps.
    """

    #: Tuple(int,int): Lower and upper possible reward (excluding limit violation reward)
    reward_range = (-np.inf, np.inf)

    def __init__(self, observed_states=None, **__):
        """
        Args:
            observed_states(list/tuple(str)): Names of the observed states. ['all'] for the observation of every state.
        """
        self._physical_system = None
        observed_states = observed_states or ()
        self._observed_states = dict.fromkeys(observed_states, 1)

        self._reference_generator = None
        self._limits = None

    def __call__(self, state, reference):
        """
        Call of the reward calculation.

        Args:
            state: State array of the environment.
            reference: Reference array of the environment.

        Returns:
            float: The reward for the state, reference pair
        """
        return self.reward(state, reference)

    def set_modules(self, physical_system, reference_generator):
        """
        Setting of the physical system, to set state arrays fitting to the environments states

        Args:
            physical_system(PhysicalSystem): The physical system of the environment
            reference_generator(ReferenceGenerator): The reference generator of the environment.
        """
        self._physical_system = physical_system
        if 'all' in self._observed_states.keys():
            self._observed_states = dict.fromkeys(physical_system.state_names, 1)

        self._observed_states = set_state_array(self._observed_states, physical_system.state_names).astype(bool)
        self._limits = physical_system.limits / abs(physical_system.limits)
        self._reference_generator = reference_generator

    def reward(self, state, reference, action=None):
        """
        Reward calculation. If limits have been violated the limit violation reward will be returned. Otherwise,
        the standard reward will be returned.

        Args:
            state(ndarray(float)): Environments state array.
            reference(ndarray(float)): Environments reference array.
            action(element of action space): The previously taken action.

        Returns:
            float: Reward for this state, reference pair.
        """
        if not self._check_limit_violation(state):
            return self._reward(state, reference, action), False
        else:
            return self._limit_violation_reward(state), True

    def reset(self, initial_state=None, initial_reference=None):
        """
        This function is called by the environment when reset.
        Inner states of the reward function can be reset here, if necessary.

        Args:
            initial_state(ndarray(float)): Initial state array of the Environment
            initial_reference(ndarray(float)): Initial reference array of the environment.
        """
        pass

    def close(self):
        """
        Called, when the environment is closed to store logs, close files etc.
        """
        pass

    def _check_limit_violation(self, state):
        """
        Check for all observed states, if limits have been violated (i.e. any (absolute) state value is greater than
        the limit defined by the physical system).

        Args:
            state(ndarray(float)): State array of the environment.

        Returns:
            bool: True, if any observed limit has been violated, False otherwise.
        """
        return (abs(state[self._observed_states]) > self._limits[self._observed_states]).any()

    def _limit_violation_reward(self, state):
        """
        Called, when limits have been violated to return a special reward for this case.

        Args:
            state(ndarray(float)): Current state array of the environment.

        Returns:
            float: The limit violation reward.
        """
        raise NotImplementedError

    def _reward(self, state, reference, action):
        """
        Standard reward function. Called, when no limits have been violated to calculate the reward based on the state
        and the reference.

        Args:
            state(ndarray(float)): State array of the environment.
            reference(ndarray(float): Reference array of the environment.
            action(element of action space): The previously taken action.

        Returns:
            float: Standard reward.
        """
        raise NotImplementedError


class PhysicalSystem:
    """
    The Physical System module encapsulates the physical model of the system as well as the simulation from one step to
    the next.
    """

    @property
    def k(self):
        """
        Returns:
             int: The current systems time step k.
        """
        return self._k

    @property
    def tau(self):
        """
        Returns:
             float: The systems time constant tau.
        """
        return self._tau

    @property
    def state_names(self):
        """
        Returns:
             ndarray(str): Array containing the names of the systems states.
        """
        return self._state_names

    @property
    def state_positions(self):
        """
        Returns:
            dict(int): Dictionary mapping the state names to its positions in the state arrays

        """
        return self._state_positions

    @property
    def action_space(self):
        """
        Returns:
            gym.Space: An OpenAI Gym Space that describes the possible actions on the system.
        """
        return self._action_space

    @property
    def state_space(self):
        """
        Returns:
             gym.Space: An OpenAI Gym Space that describes the possible states of the system.
        """
        return self._state_space

    @property
    def limits(self):
        """
        Returns:
             ndarray(float): An array containing the maximum allowed physical values for each state variable.
        """
        return NotImplementedError

    @property
    def nominal_state(self):
        """
        Returns:
             ndarray(float): An array containing the nominal values for each state variable.
        """
        return NotImplementedError

    def __init__(self, action_space, state_space, state_names, tau):
        """
        Args:
            action_space(gym.Space): The set of allowed actions on the system.
            state_space(gym.Space): The set of possible systems states.
            state_names(ndarray(str)): The names of the systems states
            tau(float): The systems simulation time interval.
        """
        self._action_space = action_space
        self._state_space = state_space
        self._state_names = state_names
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}
        self._tau = tau
        self._k = 0

    def reset(self, initial_state=None):
        """
        Reset the physical system to an initial state before a new episode starts.

        Returns:
             element of state_space: The initial systems state
        """
        raise NotImplementedError

    def simulate(self, action):
        """
        Simulation of the Physical System for one time step with the input action.
        This method is called in the environment in every step to update the systems state.

        Args:
            action(element of action_space): The action to play on the system for the next time step.

        Returns:
            element of state_space: The systems state after the action was applied.
        """
        raise NotImplementedError

    def close(self):
        """
        Called, when the environment is closed.
        Close the System and all of its submodules by closing files, saving logs etc.
        """
        pass
