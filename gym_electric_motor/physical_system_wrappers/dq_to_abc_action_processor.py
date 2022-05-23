import gym
import numpy as np

import gym_electric_motor.physical_systems as ps
from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class DqToAbcActionProcessor(PhysicalSystemWrapper):
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
        return ps.ThreePhaseMotor.t_32(ps.ThreePhaseMotor.q(action, angle))

    _registry = {}

    @classmethod
    def register_transformation(cls, motor_types):
        def wrapper(callable_):
            for motor_type in motor_types:
                cls._registry[motor_type] = callable_
        return wrapper

    @classmethod
    def make(cls, motor_type, *args, **kwargs):
        assert motor_type in cls._registry.keys(), f'Not supported motor_type {motor_type}.'
        class_ = cls._registry[motor_type]
        inst = class_(*args, **kwargs)
        return inst

    def __init__(self, angle_name, physical_system=None):
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
        self._angle_index = physical_system.state_names.index(self._angle_name)
        assert self._angle_name in physical_system.state_names, \
            f'Angle {self._angle_name} not in the states of the physical system. ' \
            f'Probably a flux observer is required.'

        self._angle_advance = 0.5

        # If dead time has been added to the system increase the angle advance by the amount of dead time steps
        if hasattr(physical_system, 'dead_time'):
            self._angle_advance += physical_system.dead_time

        return self

    def simulate(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        # Docstring of super class
        normalized_state = self._physical_system.reset()
        self._state = normalized_state * self._physical_system.limits
        return normalized_state

    def _advance_angle(self, state):
        return state[self._angle_index] \
            + self._angle_advance * self._physical_system.tau * state[self._omega_index]


class _ClassicDqToAbcActionProcessor(DqToAbcActionProcessor):

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)

    def simulate(self, action):
        # Docstring of superclass
        advanced_angle = self._advance_angle(self._state)
        abc_action = self._transformation(action, advanced_angle)
        normalized_state = self._physical_system.simulate(abc_action)
        self._state = normalized_state * self._physical_system.limits
        return normalized_state


DqToAbcActionProcessor.register_transformation(['PMSM'])(
    lambda angle_name='epsilon', *args, **kwargs: _ClassicDqToAbcActionProcessor(angle_name, *args, **kwargs)
)

DqToAbcActionProcessor.register_transformation(['SCIM'])(
    lambda angle_name='psi_angle', *args, **kwargs: _ClassicDqToAbcActionProcessor(angle_name, *args, **kwargs)
)


@DqToAbcActionProcessor.register_transformation(['DFIM'])
class _DFIMDqToAbcActionProcessor(DqToAbcActionProcessor):

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(4,))

    def __init__(self, physical_system=None):
        super().__init__('epsilon', physical_system=physical_system)
        self._flux_angle_name = 'psi_abs'
        self._flux_angle_index = None

    def simulate(self, action):
        """Dq to abc space transformation function for doubly fed induction motor environments.

        Args:
            action: The actions for the stator and rotor circuit in dq-coordinates.
        Returns:
            The next state of the physical system.
        """
        advanced_angle = self._advance_angle(self._state)
        dq_action_stator = action[:2]
        dq_action_rotor = action[2:]
        abc_action_stator = self._transformation(dq_action_stator, advanced_angle)
        abc_action_rotor = self._transformation(dq_action_rotor, self._state[self._flux_angle_index] - advanced_angle)
        abc_action = np.concatenate((abc_action_stator, abc_action_rotor))
        normalized_state = self._physical_system.simulate(abc_action)
        self._state = normalized_state * self._physical_system.limits
        return normalized_state

    def set_physical_system(self, physical_system):
        super().set_physical_system(physical_system)
        self._flux_angle_index = physical_system.state_names.index('psi_angle')
        return self
