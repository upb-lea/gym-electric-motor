import gym
import numpy as np

import gym_electric_motor.physical_systems as ps
from gym_electric_motor.state_action_processors import StateActionProcessor


class DqToAbcActionProcessor(StateActionProcessor):
    """The DqToAbcActionProcessor converts an inner system with an AC motor and actions in abc coordinates to a
    system to which actions in the dq-coordinate system can be applied.
    """

    @staticmethod
    def _transformation(action, angle):
        """Transforms the action in dq-space to an action in the abc-space for the use in the inner system.

        Args:
            action(numpy.ndarray[float]): action in the dq-space
            angle(float): Current angle of the system.

        Returns:
            numpy.ndarray[float]: The action in the abc-space
        """
        return ps.ThreePhaseMotor.t_32(ps.ThreePhaseMotor.q_inv(action, angle))

    def __init__(self, angle_name=None, physical_system=None):
        """
        Args:
            angle_name(string): Name of the state that defines the current angle of the AC-motor
            physical_system(PhysicalSystem(optional)): The inner physical system.
        """
        self._angle = 0.0
        self._angle_index = None
        self._omega_index = None
        self._state = None
        self._angle_name = angle_name
        self._angle_advance = 0.0

        super().__init__(physical_system)

    def set_physical_system(self, physical_system):
        # Docstring of super class
        assert isinstance(physical_system.electrical_motor, ps.ThreePhaseMotor), \
            'The motor in the system has to derive from the ThreePhaseMotor to define transformations.'
        super().set_physical_system(physical_system)
        self._omega_index = physical_system.state_names.index('omega')
        if isinstance(physical_system.unwrapped, ps.DoublyFedInductionMotorSystem):
            self._set_dfim_system(physical_system)
        else:
            self._action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)
            # If no angle name was passed, try to use defaults.
            # ('epsilon' for sync motors 'psi_angle' for induction motors)
            if self._angle_name is None:
                if isinstance(
                    physical_system.electrical_motor,
                    (ps.PermanentMagnetSynchronousMotor, ps.SynchronousReluctanceMotor)
                ):
                    self._angle_name = 'epsilon'
                elif isinstance(physical_system.electrical_motor, ps.SquirrelCageInductionMotor):
                    self._angle_name = 'psi_angle'
            self.simulate = self._single_angle_simulate
            self._angle_index = physical_system.state_names.index(self._angle_name)
            assert self._angle_name in physical_system.state_names, \
                f'Angle {self._angle_name} not in the states of the physical system. ' \
                f'Probably a flux observer is required.'

        self._angle_advance = 0.5

        # If dead time has been added to the system increase the angle advance by the amount of dead time steps
        if hasattr(physical_system, 'dead_time'):
            self._angle_advance += physical_system.dead_time

        return self

    def _set_dfim_system(self, physical_system):
        self._action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float64)
        self._angle_index = [
            physical_system.state_names.index('psi_angle'), physical_system.state_names.index('epsilon')
        ]
        self.simulate = self._dfim_simulate

    def simulate(self, action):
        # Docstring of super class
        raise ReferenceError('Physical System is unset.')

    def _single_angle_simulate(self, action):
        """Dq to abc space transformation function for all environments except those for a doubly fed induction motor.

        Args:
            action: The actions for the stator and rotor circuit in dq-coordinates.
        Returns:
            The next state of the physical system.
        """
        advanced_angle = self._state[self._angle_index] \
            + self._angle_advance * self._physical_system.tau * self._state[self._omega_index]
        abc_action = self._transformation(action, advanced_angle)
        normalized_state = self._physical_system.simulate(abc_action)
        self._state = normalized_state * self._physical_system.limits
        return normalized_state

    def _dfim_simulate(self, action):
        """Dq to abc space transformation function for doubly fed induction motor environments.

        Args:
            action: The actions for the stator and rotor circuit in dq-coordinates.
        Returns:
            The next state of the physical system.
        """
        advanced_angle = self._state[self._angle_index[1]] \
            + self._angle_advance * self._physical_system.tau * self._state[self._omega_index]
        dq_action_stator = action[:2]
        dq_action_rotor = action[2:]
        abc_action_stator = self._transformation(dq_action_stator, advanced_angle)
        abc_action_rotor = self._transformation(dq_action_rotor, self._angle_index[0] - advanced_angle)
        abc_action = np.concatenate((abc_action_stator, abc_action_rotor))
        normalized_state = self._physical_system.simulate(abc_action)
        self._state = normalized_state * self._physical_system.limits
        return normalized_state

    def reset(self, **kwargs):
        # Docstring of super class
        normalized_state = self._physical_system.reset()
        self._state = normalized_state * self._physical_system.limits
        return normalized_state
