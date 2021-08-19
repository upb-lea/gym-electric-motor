import numpy as np
from gym.spaces import Discrete, Box, MultiDiscrete

from ..utils import instantiate


class PowerElectronicConverter:
    """
    Base class for all converters in a SCMLSystem.
 
    Properties:
        | *voltages(tuple(float, float))*: Determines which output voltage polarities the converter can generate.
        | E.g. (0, 1) - Only positive voltages / (-1, 1) Positive and negative voltages

        | *currents(tuple(float, float))*: Determines which output current polarities the converter can generate.
        | E.g. (0, 1) - Only positive currents / (-1, 1) Positive and negative currents
    """

    #: gym.Space that defines Minimum, Maximum and Dimension of possible output voltage of the converter
    voltages = None
    #: gym.Space that defines Minimum, Maximum and Dimension of possible output current of the converter
    currents = None
    #: gym.Space that defines the set of all possible actions for the converter
    action_space = None
    #: Default action that is taken after a reset.
    _reset_action = None

    def __init__(self, tau, dead_time=False, interlocking_time=0.0):
        """
       :param tau: Discrete time step of the system in seconds
       :param dead_time: Flag, if a system dead_time of one cycle should be considered.
       :param interlocking_time: Interlocking time of the transistors in seconds
        """
        self._tau = tau
        self._dead_time = dead_time
        self._dead_time_action = self._reset_action
        self._current_action = self._reset_action
        self._interlocking_time = interlocking_time
        self._action_start_time = 0.0

    def reset(self):
        """
        Reset all converter states to a default.

        Returns:
             list(float): A default output voltage after reset(=0V).
        """
        self._dead_time_action = self._reset_action
        self._current_action = self._reset_action
        self._action_start_time = 0.0
        return [0.0]

    def set_action(self, action, t):
        """
        Set the next action of the converter at the beginning of a simulation step in the system.

        Args:
            action(element of action_space): The control action on the converter.
            t(float): Time at the beginning of the simulation step in seconds.

        Returns:
            list(float): Times when a switching action occurs and the conversion function must be called by the system.
        """
        if self._dead_time:
            self._current_action = self._dead_time_action
            self._dead_time_action = action
        else:
            self._current_action = action
        self._action_start_time = t
        return self._set_switching_pattern()

    def i_sup(self, i_out):
        """
        Calculate the current, the converter takes from the supply for the given output currents and the current
        switching state.

        Args:
            i_out(list(float)): All currents flowing out of the converter and into the motor.

        Returns:
            float: The current drawn from the supply.
        """
        raise NotImplementedError

    def convert(self, i_out, t):
        """
        The conversion function that converts the previously set action to an input voltage for the motor.
        This function has to be called at least at every previously defined switching time, because the input voltage
        for the motor might change at these times.

        Args:
            i_out(list(float)): All currents that flow out of the converter into the motor.
            t(float): Current time of the system.

        Returns:
             list(float): List of all input voltages at the motor.
        """
        raise NotImplementedError

    def _set_switching_pattern(self):
        """
        Method to calculate the switching pattern and corresponding switching times for the next time step.
        At least, the next time step [t + tau] is returned.

        Returns:
             list(float): Switching times.
        """
        self._switching_pattern = [self._current_action]
        return [self._action_start_time + self._tau]


class NoConverter(PowerElectronicConverter):
    """Dummy Converter class used to directly transfer the supply voltage to the motor"""
    # Dummy default values for voltages and currents.
    # No real use other than to fit the current physical system architecture
    voltages = Box(0, 1, shape=(3,))
    currents = Box(0, 1, shape=(3,))
    action_space = Box(low=np.array([]), high=np.array([]))

    def i_sup(self, i_out):
        return i_out[0]

    def convert(self, i_out, t):
        return [1]

class ContDynamicallyAveragedConverter(PowerElectronicConverter):
    """
    Base class for all continuously controlled converters that calculate the input voltages to the motor with a
    dynamically averaged model over one time step.

    This class also implements the interlocking time of the transistors as a discount on the output voltage.
    """

    _reset_action = [0]

    def __init__(self, tau=1e-4, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)

    def set_action(self, action, t):
        # Docstring in base class
        return super().set_action(min(max(action, self.action_space.low), self.action_space.high), t)

    def convert(self, i_out, t):
        # Docstring in base class
        return [min(max(self._convert(i_out, t) - self._interlock(i_out, t), self.voltages.low[0]), self.voltages.high[0])]

    def _convert(self, i_in, t):
        """
        Calculate an idealized output voltage for the current active action neglecting interlocking times.

        Args:
            i_in(list(float)): Input currents of the motor
            t(float): Time of the system

        Returns:
             float: Idealized output voltage neglecting interlocking times
        """
        raise NotImplementedError

    def i_sup(self, i_out):
        # Docstring in base class
        raise NotImplementedError

    def _interlock(self, i_in, *_):
        """
        Calculate the output voltage discount due to the interlocking time of the transistors

        Args:
            i_in(list(float)): list of all currents flowing into the motor.
        """
        return np.sign(i_in[0]) / self._tau * self._interlocking_time


class FiniteConverter(PowerElectronicConverter):
    """
    Base class for all finite converters.
    """

    #: The switching states of the converter for the current action
    _switching_pattern = []
    #: The current switching state of the converter
    _switching_state = 0
    #: The action that is the default after reset
    _reset_action = 0

    def __init__(self, tau=1e-5, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)

    def set_action(self, action, t):
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        return super().set_action(action, t)

    def convert(self, i_out, t):
        # Docstring in base class
        raise NotImplementedError

    def i_sup(self, i_out):
        # Docstring in base class
        raise NotImplementedError


class FiniteOneQuadrantConverter(FiniteConverter):
    """
    Key:
        'Finite-1QC'

    Switching States / Actions:
        | 0: Transistor off.
        | 1: Transistor on.

    Action Space:
        Discrete(2)

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(0, 1, shape=(1,))
    """

    voltages = Box(0, 1, shape=(1,))
    currents = Box(0, 1, shape=(1,))
    action_space = Discrete(2)

    def convert(self, i_out, t):
        # Docstring in base class
        return [self._current_action if i_out[0] >= 0 else 1]

    def i_sup(self, i_out):
        # Docstring in base class
        return i_out[0] if self._current_action == 1 else 0


class FiniteTwoQuadrantConverter(FiniteConverter):
    """
    Key:
        'Finite-2QC'

    Switching States / Actions:
        | 0: Both Transistors off.
        | 1: Upper Transistor on.
        | 2: Lower Transistor on.

    Action Space:
        Discrete(3)

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(-1, 1, shape=(1,))
    """

    voltages = Box(0, 1, shape=(1,))
    currents = Box(-1, 1, shape=(1,))
    action_space = Discrete(3)

    def convert(self, i_out, t):
        # Docstring in base class
        # Converter switches slightly (tau / 1000 seconds) before interlocking time due to inaccuracy of the solvers.
        if t - self._tau / 1000 > self._action_start_time + self._interlocking_time:
            self._switching_state = self._switching_pattern[-1]
        else:
            self._switching_state = self._switching_pattern[0]
        if self._switching_state == 0:
            if i_out[0] < 0:
                return [1]
            elif i_out[0] >= 0:
                return [0.0]
        elif self._switching_state == 1:
            return [1]
        elif self._switching_state == 2:
            return [0.0]
        else:
            raise Exception('Invalid switching state of the converter')

    def i_sup(self, i_out):
        # Docstring in base class
        if self._switching_state == 0:
            return i_out[0] if i_out[0] < 0 else 0
        elif self._switching_state == 1:
            return i_out[0]
        elif self._switching_state == 2:
            return 0
        else:
            raise Exception('Invalid switching state of the converter')

    def _set_switching_pattern(self):
        # Docstring in base class
        if (
                self._current_action == 0
                or self._switching_state == 0
                or self._current_action == self._switching_state
                or self._interlocking_time == 0
        ):
            self._switching_pattern = [self._current_action]
            return [self._action_start_time + self._tau]
        else:
            self._switching_pattern = [0, self._current_action]
            return [self._action_start_time + self._interlocking_time, self._action_start_time + self._tau]


class FiniteFourQuadrantConverter(FiniteConverter):
    """
    Key:
        'Finite-4QC'

    Switching States / Actions:
        | 0: T2, T4 on.
        | 1: T1, T4 on.
        | 2: T2, T3 on.
        | 3: T1, T3 on.

    Action Space:
        Discrete(4)

    Output Voltages and Currents:
        | Box(-1, 1, shape=(1,))
        | Box(-1, 1, shape=(1,))
    """
    voltages = Box(-1, 1, shape=(1,))
    currents = Box(-1, 1, shape=(1,))
    action_space = Discrete(4)

    def __init__(self, **kwargs):
        # Docstring in base class
        super().__init__(**kwargs)
        self._subconverters = [FiniteTwoQuadrantConverter(**kwargs), FiniteTwoQuadrantConverter(**kwargs)]

    def reset(self):
        # Docstring in base class
        self._subconverters[0].reset()
        self._subconverters[1].reset()
        return super().reset()

    def convert(self, i_out, t):
        # Docstring in base class
        return [self._subconverters[0].convert(i_out, t)[0] - self._subconverters[1].convert([-i_out[0]], t)[0]]

    def set_action(self, action, t):
        # Docstring in base class
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        times = []
        action0 = [1, 1, 2, 2][action]
        action1 = [1, 2, 1, 2][action]
        times += self._subconverters[0].set_action(action0, t)
        times += self._subconverters[1].set_action(action1, t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        return self._subconverters[0].i_sup(i_out) + self._subconverters[1].i_sup([-i_out[0]])


class ContOneQuadrantConverter(ContDynamicallyAveragedConverter):
    """
    Key:
        'Cont-1QC'

    Action:
        Duty Cycle of the Transistor in [0,1].

    Action Space:
        Box([0,1])

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(0, 1, shape=(1,))
    """
    voltages = Box(0, 1, shape=(1,))
    currents = Box(0, 1, shape=(1,))
    action_space = Box(0, 1, shape=(1,))

    def _convert(self, i_in, *_):
        # Docstring in base class
        return self._current_action[0] if i_in[0] >= 0 else 1

    def _interlock(self, *_):
        # Docstring in base class
        return 0

    def i_sup(self, i_out):
        # Docstring in base class
        return self._current_action[0] * i_out[0]


class ContTwoQuadrantConverter(ContDynamicallyAveragedConverter):
    """
    Key:
        'Cont-2QC'

    Actions:
        | Duty Cycle upper Transistor: Action
        | Duty Cycle upper Transistor: 1 - Action

    Action Space:
        Box([0,1])

    Output Voltages and Currents:
        | voltages: Box(0, 1, shape=(1,))
        | currents: Box(-1, 1, shape=(1,))
    """
    voltages = Box(0, 1, shape=(1,))
    currents = Box(-1, 1, shape=(1,))
    action_space = Box(0, 1, shape=(1,))


    def _convert(self, *_):
        # Docstring in base class
        return self._current_action[0]

    def i_sup(self, i_out):
        # Docstring in base class
        interlocking_current = 1 if i_out[0] < 0 else 0
        return (
            self._current_action[0]
            + self._interlocking_time / self._tau * (interlocking_current - self._current_action[0])
        ) * i_out[0]


class ContFourQuadrantConverter(ContDynamicallyAveragedConverter):
    """
    The continuous four quadrant converter (4QC) is simulated with two continuous 2QC.

    Key:
        'Cont-4QC'

    Actions:
        | Duty Cycle Transistor T1: 0.5 * (Action + 1)
        | Duty Cycle Transistor T2: 1 - 0.5 * (Action + 1)
        | Duty Cycle Transistor T3: 1 - 0.5 * (Action + 1)
        | Duty Cycle Transistor T4: 0.5 * (Action + 1)

    Action Space:
        Box(-1, 1, shape=(1,))

    Output Voltages and Currents:
        | voltages: Box(-1, 1, shape=(1,))
        | currents: Box(-1, 1, shape=(1,))
    """
    voltages = Box(-1, 1, shape=(1,))
    currents = Box(-1, 1, shape=(1,))
    action_space = Box(-1, 1, shape=(1,))

    def __init__(self, **kwargs):
        # Docstring in base class
        super().__init__(**kwargs)
        self._subconverters = [ContTwoQuadrantConverter(**kwargs), ContTwoQuadrantConverter(**kwargs)]

    def _convert(self, *_):
        # Not used here
        pass

    def reset(self):
        # Docstring in base class
        self._subconverters[0].reset()
        self._subconverters[1].reset()
        return super().reset()

    def convert(self, i_out, t):
        # Docstring in base class
        return [self._subconverters[0].convert(i_out, t)[0] - self._subconverters[1].convert(i_out, t)[0]]

    def set_action(self, action, t):
        # Docstring in base class
        super().set_action(action, t)
        times = []
        times += self._subconverters[0].set_action([0.5 * (action[0] + 1)], t)
        times += self._subconverters[1].set_action([-0.5 * (action[0] - 1)], t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        return self._subconverters[0].i_sup(i_out) + self._subconverters[1].i_sup([-i_out[0]])


class FiniteMultiConverter(FiniteConverter):
    """
    Converter that allows to include an arbitrary number of independent finite subconverters.
    Subconverters must be 'elementary' and can not be MultiConverters.

    Key:
        'Finite-Multi'

    Actions:
        Concatenation of the subconverters' action spaces

    Action Space:
        MultiDiscrete([subconverter[0].action_space.n , subconverter[1].action_space.n, ...])

    Output Voltage Space:
        Box([subconverter[0].voltages.low, subconverter[1].voltages.low, ...],
            [subconverter[0].voltages.high, subconverter[1].voltages.high, ...])
    """

    @property
    def subconverters(self):
        return self._subconverters

    def __init__(self, subconverters, **kwargs):
        """
        Args:
            subconverters(list(str/class/object): Subconverters to instantiate .
            kwargs(dict): Parameters to pass to the Subconverters and the superclass
        """
        super().__init__(**kwargs)
        self._subconverters = [
            instantiate(PowerElectronicConverter, subconverter, **kwargs) for subconverter in subconverters
        ]
        self.subsignal_current_space_dims = []
        self.subsignal_voltage_space_dims = []
        self.action_space = []
        currents_low = []
        currents_high = []
        voltages_low = []
        voltages_high = []

        # get the limits and space dims from each subconverter
        for subconverter in self._subconverters:
            self.subsignal_current_space_dims.append(np.squeeze(subconverter.currents.shape) or 1)
            self.subsignal_voltage_space_dims.append(np.squeeze(subconverter.voltages.shape) or 1)

            self.action_space.append(subconverter.action_space.n)

            currents_low.append(subconverter.currents.low)
            currents_high.append(subconverter.currents.high)

            voltages_low.append(subconverter.voltages.low)
            voltages_high.append(subconverter.voltages.high)

        # convert to 1D list
        self.subsignal_current_space_dims = np.array(self.subsignal_current_space_dims)
        self.subsignal_voltage_space_dims = np.array(self.subsignal_voltage_space_dims)

        currents_low = np.concatenate(currents_low)
        currents_high = np.concatenate(currents_high)

        voltages_low = np.concatenate(voltages_low)
        voltages_high = np.concatenate(voltages_high)

        # put limits into gym_space format
        self.action_space = MultiDiscrete(self.action_space)
        self.currents = Box(currents_low, currents_high)
        self.voltages = Box(voltages_low, voltages_high)

    def convert(self, i_out, t):
        # Docstring in base class
        u_in = []
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._subconverters, self.subsignal_voltage_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            u_in += subconverter.convert(i_out[subsignal_idx_low:subsignal_idx_high], t)
            subsignal_idx_low = subsignal_idx_high
        return u_in

    def reset(self):
        # Docstring in base class
        u_in = []
        for subconverter in self._subconverters:
            u_in += subconverter.reset()
        return u_in

    def set_action(self, action, t):
        # Docstring in base class
        times = []
        for subconverter, sub_action in zip(self._subconverters, action):
            times += subconverter.set_action(sub_action, t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        i_sup = 0
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._subconverters, self.subsignal_current_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            i_sup += subconverter.i_sup(i_out[subsignal_idx_low:subsignal_idx_high])
            subsignal_idx_low = subsignal_idx_high
        return i_sup


class ContMultiConverter(ContDynamicallyAveragedConverter):
    """
    Converter that allows to include an arbitrary number of independent continuous subconverters.
    Subconverters must be 'elementary' and can not be MultiConverters.

    Key:
        'Cont-Multi'

    Actions:
        Concatenation of the subconverters' action spaces

    Action Space:
        Box([subconverter[0].action_space.low, subconverter[1].action_space.low, ...],
            [subconverter[0].action_space.high, subconverter[1].action_space.high, ...])

    Output Voltage Space:
        Box([subconverter[0].voltages.low, subconverter[1].voltages.low, ...],
            [subconverter[0].voltages.high, subconverter[1].voltages.high, ...])
    """

    def __init__(self, subconverters, **kwargs):
        """
        Args:
            subconverters(list(str/class/object): Subconverters to instantiate .
            kwargs(dict): Parameters to pass to the Subconverters
        """
        super().__init__(**kwargs)
        self._subconverters = [instantiate(PowerElectronicConverter, subconverter, **kwargs) for subconverter in subconverters]

        self.subsignal_current_space_dims = []
        self.subsignal_voltage_space_dims = []
        action_space_low = []
        action_space_high = []
        currents_low = []
        currents_high = []
        voltages_low = []
        voltages_high = []

        # get the limits and space dims from each subconverter
        for subconverter in self._subconverters:
            self.subsignal_current_space_dims.append(np.squeeze(subconverter.currents.shape) or 1)
            self.subsignal_voltage_space_dims.append(np.squeeze(subconverter.voltages.shape) or 1)

            action_space_low.append(subconverter.action_space.low)
            action_space_high.append(subconverter.action_space.high)

            currents_low.append(subconverter.currents.low)
            currents_high.append(subconverter.currents.high)

            voltages_low.append(subconverter.voltages.low)
            voltages_high.append(subconverter.voltages.high)

        # convert to 1D list
        self.subsignal_current_space_dims = np.array(self.subsignal_current_space_dims)
        self.subsignal_voltage_space_dims = np.array(self.subsignal_voltage_space_dims)

        action_space_low = np.concatenate(action_space_low)
        action_space_high = np.concatenate(action_space_high)

        currents_low = np.concatenate(currents_low)
        currents_high = np.concatenate(currents_high)

        voltages_low = np.concatenate(voltages_low)
        voltages_high = np.concatenate(voltages_high)

        # put limits into gym_space format
        self.action_space = Box(action_space_low, action_space_high)
        self.currents = Box(currents_low, currents_high)
        self.voltages = Box(voltages_low, voltages_high)

    def set_action(self, action, t):
        # Docstring in base class
        times = []
        ind = 0
        for subconverter in self._subconverters:
            sub_action = action[ind:ind + subconverter.action_space.shape[0]]
            ind += subconverter.action_space.shape[0]
            times += subconverter.set_action(sub_action, t)
        return sorted(list(set(times)))

    def reset(self):
        # Docstring in base class
        u_in = []
        for subconverter in self._subconverters:
            u_in += subconverter.reset()
        return u_in

    def convert(self, i_out, t):
        # Docstring in base class
        u_in = []
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._subconverters, self.subsignal_voltage_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            u_in += subconverter.convert(i_out[subsignal_idx_low:subsignal_idx_high], t)
            subsignal_idx_low = subsignal_idx_high
        return u_in

    def _convert(self, i_in, t):
        # Not used
        pass

    def i_sup(self, i_out):
        # Docstring in base class
        i_sup = 0
        subsignal_idx_low = 0
        for subconverter, subsignal_space_size in zip(self._subconverters, self.subsignal_current_space_dims):
            subsignal_idx_high = subsignal_idx_low + subsignal_space_size
            i_sup += subconverter.i_sup(i_out[subsignal_idx_low:subsignal_idx_high])
            subsignal_idx_low = subsignal_idx_high

        return i_sup


class FiniteB6BridgeConverter(FiniteConverter):
    """
    The finite B6 bridge converters (B6C) is simulated with three finite 2QC.

    Key:
        'Finite-B6C'

    Actions:
        +-+-----+-----+-----+
        | |H_1  |H_2  |H_3  |
        +=+=====+=====+=====+
        |0|lower|lower|lower|
        +-+-----+-----+-----+
        |1|lower|lower|upper|
        +-+-----+-----+-----+
        |2|lower|upper|lower|
        +-+-----+-----+-----+
        |3|lower|upper|upper|
        +-+-----+-----+-----+
        |4|upper|lower|lower|
        +-+-----+-----+-----+
        |5|upper|lower|upper|
        +-+-----+-----+-----+
        |6|upper|upper|lower|
        +-+-----+-----+-----+
        |7|upper|upper|upper|
        +-+-----+-----+-----+

    Action Space:
        Discrete(8)

    Output Voltages and Currents:
        | voltages: Box(-1,1, shape=(3,))
        | currents: Box(-1,1, shape=(3,))

    Output Voltage Space:
        Box(-0.5, 0.5, shape=(3,))
    """

    action_space = Discrete(8)
    # Only positive voltages can be applied
    voltages = Box(-1, 1, shape=(3,))
    # positive and negative currents are possible
    currents = Box(-1, 1, shape=(3,))
    _reset_action = 0
    _subactions = [
        [2, 2, 2],
        [2, 2, 1],
        [2, 1, 2],
        [2, 1, 1],
        [1, 2, 2],
        [1, 2, 1],
        [1, 1, 2],
        [1, 1, 1]
    ]

    def __init__(self, tau=1e-5, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)
        self._subconverters = [
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
            FiniteTwoQuadrantConverter(tau=tau, **kwargs),
        ]

    def reset(self):
        # Docstring in base class
        return [
            self._subconverters[0].reset()[0] - 0.5,
            self._subconverters[1].reset()[0] - 0.5,
            self._subconverters[2].reset()[0] - 0.5,
        ]

    def convert(self, i_out, t):
        # Docstring in base class
        u_out = [
            self._subconverters[0].convert([i_out[0]], t)[0] - 0.5,
            self._subconverters[1].convert([i_out[1]], t)[0] - 0.5,
            self._subconverters[2].convert([i_out[2]], t)[0] - 0.5
        ]
        return u_out

    def set_action(self, action, t):
        # Docstring in base class
        assert self.action_space.contains(action), \
            f"The selected action {action} is not a valid element of the action space {self.action_space}."
        subactions = self._subactions[action]
        times = []
        times += self._subconverters[0].set_action(subactions[0], t)
        times += self._subconverters[1].set_action(subactions[1], t)
        times += self._subconverters[2].set_action(subactions[2], t)
        return sorted(list(set(times)))

    def i_sup(self, i_out):
        # Docstring in base class
        return sum([subconverter.i_sup([i_out_]) for subconverter, i_out_ in zip(self._subconverters, i_out)])


class ContB6BridgeConverter(ContDynamicallyAveragedConverter):
    """
    The continuous B6 bridge converter (B6C) is simulated with three continuous 2QC.

    Key:
        'Cont-B6C'

    Actions:
        The Duty Cycle for each half bridge in the range of (-1,1)

    Action Space:
        Box(-1, 1, shape=(3,))

    Output Voltages and Currents:
        | voltages: Box(-1,1, shape=(3,))
        | currents: Box(-1,1, shape=(3,))

    Output Voltage Space:
        Box(-0.5, 0.5, shape=(3,))
    """

    action_space = Box(-1, 1, shape=(3,))
    # Only positive voltages can be applied
    voltages = Box(-1, 1, shape=(3,))
    # Positive and negative currents are possible
    currents = Box(-1, 1, shape=(3,))

    _reset_action = [0, 0, 0]

    def __init__(self, tau=1e-4, **kwargs):
        # Docstring in base class
        super().__init__(tau=tau, **kwargs)
        self._subconverters = [
            ContTwoQuadrantConverter(tau=tau, **kwargs),
            ContTwoQuadrantConverter(tau=tau, **kwargs),
            ContTwoQuadrantConverter(tau=tau, **kwargs),
        ]

    def reset(self):
        # Docstring in base class
        return [
            self._subconverters[0].reset()[0] - 0.5,
            self._subconverters[1].reset()[0] - 0.5,
            self._subconverters[2].reset()[0] - 0.5,
        ]

    def convert(self, i_out, t):
        # Docstring in base class
        u_out = [
            self._subconverters[0].convert([i_out[0]], t)[0] - 0.5,
            self._subconverters[1].convert([i_out[1]], t)[0] - 0.5,
            self._subconverters[2].convert([i_out[2]], t)[0] - 0.5
        ]
        return u_out

    def set_action(self, action, t):
        # Docstring in base class
        times = []
        times += self._subconverters[0].set_action([0.5 * (action[0] + 1)], t)
        times += self._subconverters[1].set_action([0.5 * (action[1] + 1)], t)
        times += self._subconverters[2].set_action([0.5 * (action[2] + 1)], t)
        return sorted(list(set(times)))

    def _convert(self, i_in, t):
        # Not used
        pass

    def i_sup(self, i_out):
        # Docstring in base class
        return sum([subconverter.i_sup([i_out_]) for subconverter, i_out_ in zip(self._subconverters, i_out)])
