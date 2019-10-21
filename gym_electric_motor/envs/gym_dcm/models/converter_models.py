from gym import spaces
import numpy as np


class Converter(object):
    """
    Class that is modeling DC converter. It can be parametrized to the following converters
    by passing type='{disc,cont}-{1Q,2Q,4Q}' :
                            disc_act   / cont_act           / returns_cont:\n
        1 Quadrant: actions: 0, 1      / [0, 1]             / [0, 1]\n
        2 Quadrant: actions: 0, 1, 2   / [0, 1]             / [0, 1]\n
        4 Quadrant: actions: 0, 1, 2, 3/ [-1 , 1]           / [-1, 1]\n

    The specific converter also specifies the action space of the surrounding gym environment.
    Call convert(action, i_in) to calculate the normalised current output voltage at the converter.
    The return value is limited due to the constraints. Converter can consider a fixed dead time of one time step.
    If this option is selected, the action will be applied in the subsequent time step.

    cont:
        In this case dynamic averaging is applied and assumed that the converter works ideally except interlocking time.
        Due to this the output voltage is the same as the input voltage and the added voltage error of the interlocking.
        The interlocking has got only an effect at the 2Q converter.

    disc:
        In this case the action includes the switching commands for the transistors and the converter determines the
        output voltage, depending on the last switching state, input current and action. The interlocking time is
        considered as well. It is not possible to switch between every switching states instantaneously and due to this,
        further states are included. This is described more detailed in the subclass of the 4Q Converter.

    1Q: It consists of one transistor and at least on diode and only positive voltages and currents are possible.\n
    2Q: It is an asymmetric half-bridge, such that only positive voltages and currents in both directions are possible.\n
    4Q: Two half-bridges are in parallel and voltage and currents can be positive and negative.\n
    """

    # The converter determines the action space of the environment
    action_space = None
    # The possible voltages (low, high) this converter can generate at the input terminals of the motor
    # If voltages[0] == 0 the converter cannot invert the voltages. If voltages[0] == -1 it can.
    voltages = (None, None)
    # The possible currents (low, high) this converter can generate at the input terminals of the motor
    # If currents[0] == 0 the converter cannot invert the currents. If currents[0] == -1 it is possible.
    currents = (None, None)

    @classmethod
    def make(cls, converter_type, tau, interlocking_time=0.0, dead_time=True):
        """
        Factory class method for generating a converter.

        Args:
            converter_type:  Type of the converter as string.
            tau: Cycle time of the environment
            interlocking_time: interlocking time of the converter that it takes to set a voltage at the output
            dead_time: specifies if a dead time will be considered

        Returns:
            An initialized converter instance of the described type.
        """
        model = {
            'disc-1Q': DiscreteOneQuadrantConverter,
            'disc-2Q': DiscreteTwoQuadrantConverter,
            'disc-4Q': DiscreteFourQuadrantConverter,
            'cont-1Q': ContOneQuadrantConverter,
            'cont-2Q': ContTwoQuadrantConverter,
            'cont-4Q': ContFourQuadrantConverter
        }[converter_type]
        return model(interlocking_time, tau, dead_time)

    def __init__(self, interlocking_time, tau, dead_time):
        """
        General converter constructor.

        Basic components tau and the interlocking_time are set.
        The dead_time_action contains the action of the previous time step.

        Args:
            interlocking_time: interlocking time of the Converter between the switches of the Transistors
            tau: Time constant of the whole system
            dead_time: Specifies if a dead time of one time step is considered
        """

        self._interlocking_time = max(0, interlocking_time)
        self._tau = tau
        self._dead_time = dead_time
        self._dead_time_action = 0
        self._time_last_call = 0  # time of last call of convert  to consider interlocking time
        self._current_action = 0  # currently applied action

    def convert(self, action, i_in, t):
        """
        The concrete conversion function of the converter.

        Call this function and the selected conversion function will be called.
        It is implemented by the subclasses

        Args:
            action: The selected action by the agent.
            i_in: The current flowing into the motor.
            t: The time of the system

        Returns:
            The concrete conversion function
        """
        raise NotImplementedError

    @property
    def interlocking_time(self):
        """
        Returns:
            interlocking time of the Converter in seconds
        """
        return self._interlocking_time

    @interlocking_time.setter
    def interlocking_time(self, t):
        """
        interlocking Time > 0 in seconds
        """
        self._interlocking_time = max(0, t)

    def determine_current_action_disc(self, action, t):
        """
        This function determines the applied action based on the current and previous action and the dead time for
        discrete converter.
        If the current time is larger than at the last call plus the interlocking time, the next sampling step is
        reached and the action of the previous time steps needs to be applied.

        Args:
            action: The selected action by the agent
            t: current time of the motor
        """
        if self._dead_time:
            if t > self._time_last_call + self._interlocking_time:
                self._current_action = self._dead_time_action
                self._dead_time_action = action
            self._time_last_call = t
        else:
            self._current_action = action

    def determine_current_action_cont(self, action):
        """
        This function determines the applied action based on the current and previous action and the dead time for
        continuous converter.
        If a dead time is considered the action will be applied one step later.

        Args:
            action: The selected action by the agent

        """
        if self._dead_time:
            self._current_action = self._dead_time_action
            self._dead_time_action = action
        else:
            self._current_action = action


class DiscreteOneQuadrantConverter(Converter):
    """
    A discrete one quadrant converter (1QC).\n
    Actions: {0,1}\n
    For further information on all converters, look at the Converter Base class
    """

    action_space = spaces.Discrete(2)
    # Only positive voltages can be applied
    voltages = (0, 1)
    # Only positive currents
    currents = (0, 1)

    def convert(self, action, i_in, t):
        """
        Function to calculate the effective voltage at the output of a discrete controlled 1QC.

        Calculates the resulting input voltage at the input terminals of a DC motor by using a one quadrant converter.
        The action space is 0,1\n
        action = 0: Transistor is off\n
        action = 1: Transistor is on\n

        A dead time of one sampling time step can be considered.

        Args:
            action: Valid values are 0 , 1
            i_in: output current of the converter and input current of the motor
            t: current time

        Returns:
            The resulting voltage at the input of the following circuit
        """
        self.determine_current_action_disc(action, t)
        return self._current_action if i_in >= 0 else 1


class ContOneQuadrantConverter(Converter):
    """
    A continuous one quadrant converter.\n
    Actions: [0,1]\n
    For further information on all converters, look at the Converter Base class
    """

    action_space = spaces.Box(0, 1, shape=(1,))
    # Only positive voltages can be applied
    voltages = (0, 1)
    # Only positive currents
    currents = (0, 1)

    def convert(self, action, i_in, *_):
        """
        Function to calculate the effective voltage at the output of a continuously controlled 1QC.

        Calculates the resulting input voltage at the input terminals of a DC motor by using a buck converter.
        The action space is [0,1], so only positive voltages are possible.
        A dead time of one sampling time step can be considered.

        Args:
            action: normalized desired voltage with respect to the supply voltage
            i_in:output current of the converter and input current of the motor, which must be positive

        Returns:
            The resulting voltage at the input of the DC motor
        """
        self.determine_current_action_cont(action)
        return self._current_action if i_in >= 0 else 1


class DiscreteTwoQuadrantConverter(Converter):
    """
    Class models the 2Q Converter as an asymmetric half bridge. Actions are the switching commands.\n
    Actions: {0, 1, 2}\n
    For further information on all converters, look at the Converter Base class
    """
    action_space = spaces.Discrete(3)
    # Only positive voltages can be applied
    voltages = (0, 1)
    # Positive and negative currents
    currents = (-1, 1)

    def __init__(self, interlocking_time, tau, dead_time):
        """
        General converter constructor.

        Basic components tau and the interlocking_time are set.

        Args:
            interlocking_time: Interlocking time of the Converter between the switches of the Transistors
            tau: Time constant of the whole system
            dead_time: specifies if a dead time will be considered
        """
        super().__init__(interlocking_time, tau, dead_time)
        self._last_state = 0

    def convert(self, action, i_in, t):
        """
        Function to calculate the effective voltage at the output of a discrete controlled 2Q-Converter.

        Calculates the resulting input voltage at the input terminals of a DC motor by using a two quadrant converter
        without pulse width modulation.\n
        The actions are the discrete states of transistor 1 and 2.\n
        0: no transistor active, both are blocking\n
        1: Transistor 1 conducting\n
        2: Transistor 2 conducting\n
        Both transistors active would result in an short circuit and is illegal
        A dead time of one sampling time step can be considered.

        Args:
            action: Valid values are 0 (T1 off ,T2 off), 1 (T1 on, T2 off), 2 (T1 off, T2 on)
            i_in: output current of the converter and input current of the motor
            t: current time

        Returns:
            The resulting voltage at the input of the following circuit
        """

        self.determine_current_action_disc(action, t)
        if self._last_state == 0 or self._current_action == self._last_state or self._current_action == 0:
            next_action = self._current_action
        else:
            next_action = 0

        self._last_state = next_action
        if next_action == 0:
            if i_in < 0:
                return 1
            elif i_in >= 0:
                return 0.0
        if next_action == 1:
            return 1
        if next_action == 2:
            return 0.0


class ContTwoQuadrantConverter(Converter):
    """
    Class models the 2QC as an asymmetric half bridge. The action is the desired normalised input voltage
    for the motor.
    """
    action_space = spaces.Box(0, 1, shape=(1,))

    # Only positive voltages can be applied
    voltages = (0, 1)

    # Positive and negative currents
    currents = (-1, 1)

    def convert(self, action, i_in, *_):
        """
        Function to calculate the effective voltage at the output of a continuously controlled 2QC.

        Calculates the resulting input voltage at the input terminals of a DC motor by using an ideal
        two quadrant converter.
        The action space is [0 , 1]. Only positive voltages but bidirectional currents are possible.
        A dead time of one sampling time step can be considered.

        Args:
            action: Valid values are in the range [0, 1] as a normalized desired voltage.
            i_in: output current of the converter and input current of the motor

        Returns:
            The resulting average voltage at the input of the following circuit
        """
        self.determine_current_action_cont(action)
        return self._current_action - np.sign(i_in) / self._tau * self._interlocking_time


class DiscreteFourQuadrantConverter(Converter):
    """
    4QC, consisting of two parallel half bridges
    The state space of the converter is larger, because it includes also the cases, when only one or no transistor
    is conducting. The definition of the states is given below. '1' means conducting and '0' is switched of. The
    first four (0-3) states are the actions of the action space.\n
    T1 (upper left), T2( lower left), T3 (upper right), T4 (lower right)\n

    +--+--+--+--+--+
    |  |T1|T2|T3|T4|
    +==+==+==+==+==+
    |0:|1 |0 |1 |0 |
    +--+--+--+--+--+
    |1:|1 |0 |0 |1 |
    +--+--+--+--+--+
    |2:|0 |1 |1 |0 |
    +--+--+--+--+--+
    |3:|0 |1 |0 |1 |
    +--+--+--+--+--+
    |4:|0 |0 |0 |0 |
    +--+--+--+--+--+
    |5:|1 |0 |0 |0 |
    +--+--+--+--+--+
    |6:|0 |1 |0 |0 |
    +--+--+--+--+--+
    |7:|0 |0 |1 |0 |
    +--+--+--+--+--+
    |8:|0 |0 |0 |1 |
    +--+--+--+--+--+

    The switching table includes all transitions during the switching. The last state specifies the row and the
    column and the column is specified by the next action. The elements in the table define states that needs to be
    applied if you want to go from the last state to the next state. Due to the interlocking time, this can not
    always be done directly.
    """
    action_space = spaces.Discrete(4)

    # Positive and negative voltages can be applied
    voltages = (-1, 1)

    # Positive and negative currents
    currents = (-1, 1)

    def __init__(self, interlocking_time, tau, dead_time):
        """
        Initialisation of the discrete 4QC is special because further transformation matrices to consider the
        interlocking time needs to be setup. More details can be found in the init of the base class.

        Args:
            interlocking_time: Interlocking time of the Converter between the switches of the Transistors
            tau: Time constant of the whole system
            dead_time: specifies if a dead time will be considered
        """
        super().__init__(interlocking_time, tau, dead_time)
        self._last_state = 4
        # Defines the transitions which are necessary between to states and if there is an interlocking time necessary
        # From row to column
        self._switching_table = np.array([[0, 5, 7, 4, 4, 5, 4, 7, 8],
                                          [5, 1, 4, 8, 4, 5, 4, 7, 8],
                                          [7, 4, 2, 6, 4, 4, 6, 7, 4],
                                          [4, 8, 6, 3, 4, 4, 6, 4, 8],
                                          [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                          [0, 1, 4, 4, 4, 5, 4, 7, 8],
                                          [4, 4, 2, 3, 4, 4, 6, 7, 8],
                                          [0, 4, 2, 4, 4, 5, 6, 7, 4],
                                          [0, 4, 4, 3, 4, 5, 6, 4, 8]])

        # Defines the input voltage row selected by the state and the column 0 if i_in>0 and 1 else
        self._input_voltage = np.array([[0, 0],
                                        [1, 1],
                                        [-1, -1],
                                        [0, 0],
                                        [-1, 1],
                                        [0, 1],
                                        [-1, 0],
                                        [-1, 0],
                                        [0, 1]])

    def convert(self, action, i_in, t):
        """
        Function to calculate the effective voltage at the output of a discrete controlled 4QC.

        Calculates the resulting input voltage at the input terminals of a DC motor by using an ideal
        four quadrant converter. The Transistors in one branch are switched complementary, so T1 and T3 define
        the states of T2 and T4.\n
        The discrete action space is {0,1,2,3}.\n
        A dead time of one sampling time step can be considered.

        Args:
            action: Valid values are {0,1,2,3} for the switching states.
            i_in: output current of the converter and input current of the motor
            t: current time

        Returns:
            The resulting average voltage at the input of the following circuit
        """

        self.determine_current_action_disc(action, t)
        next_action = self._switching_table[self._last_state, self._current_action]
        self._last_state = next_action
        temp_val_i_in = 0 if i_in > 0 else 1
        return self._input_voltage[next_action, temp_val_i_in]


class ContFourQuadrantConverter(Converter):
    """
    4QC, consisting of two parallel half bridges
    The action is the desired input voltage for the motor.
    """
    action_space = spaces.Box(-1, 1, shape=(1,))

    # Positive and negative voltages can be applied
    voltages = (-1, 1)

    # Positive and negative currents can flow
    currents = (-1, 1)

    def convert(self, action, i_in, *_):
        """
        Function to calculate the effective voltage at the output of a continuously controlled 4QC.

        Calculates the resulting input voltage at the input terminals of a DC motor by using an ideal
        four quadrant converter.
        The action space is [-1, 1] and positive and negative voltages and currents are possible.

        A dead time of one sampling time step can be considered.

        Args:
            action: Valid values are in the range [-1, 1] as a normalized desired voltage.
            i_in: output current of the converter and input current of the motor

        Returns:
            The resulting average voltage at the input of the following circuit
        """
        self.determine_current_action_cont(action)
        return max(-1, min(1, self._current_action - 2 * np.sign(i_in) * self._tau * self._interlocking_time))
