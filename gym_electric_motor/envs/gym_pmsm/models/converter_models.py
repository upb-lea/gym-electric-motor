import numpy as np
from gym import spaces


class Disc2Level3PhaseConverter:
    """
    Converter for switching sequences. From DC to three phases ac with a B6 bridge and transistors are used such that
    three half-bridges are in parallel. 8 different switching states are possible with the following meaning:

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

    A lower position means -u_dc/2 and upper position means +u_dc/2. Because of interlocking times further states are
    used. Ech half-bridge (H_1, H_2, H_3) can be interpreted as an asymmetric 2 Quadrant Converter (positive and
    negative current, positive voltage) with four states each:

    +--+-----+--------+------------------------------------------+-------------------------------------------+
    |  |  T_1|     T_2|     u_out                                |comments                                   |
    +==+=====+========+==========================================+===========================================+
    |00|  OFF|     OFF| +u_dc/2 if i_in >0 and -u_dc/2 if i_in<0 |always possible, used as interlocking state|
    +--+-----+--------+------------------------------------------+-------------------------------------------+
    |01|  OFF|     ON |    -u_dc/2                               |lower Transistor ON                        |
    +--+-----+--------+------------------------------------------+-------------------------------------------+
    |02|  ON |     OFF|     +u_dc/2                              |upper Transistor ON                        |
    +--+-----+--------+------------------------------------------+-------------------------------------------+
    |03|  ON |     ON |     NOT ALLOWED state, short circuit.    |This state is never used                   |
    +--+-----+--------+------------------------------------------+-------------------------------------------+

    Each half-bridge is computed on its own. Lower position of the half-bridge means state 01 and upper position means
    state 02. The 00 state is only used in interlocking times.

    A symmetric output voltage range from -u_dc/2 to +u_dc/2 is used with the supply voltage u_dc and an imaginary zero
    reference point in the center for the converter, such that this shifted voltage range can be used.
    A dead time of one time step can be considered.
    """

    action_space = spaces.Discrete(8)
    # Only positive voltages can be applied
    voltages = (-1, 1)
    # positive and negative currents are possible
    currents = (-1, 1)

    @property
    def interlocking_time(self):
        """
        Returns:
            interlocking time of the Converter in seconds
        """
        return self._interlocking_time

    def __init__(self, interlocking_time=0, tau=1e-5, dead_time=True):
        """
        Basic Setting of Converter

        Args:
            interlocking_time: interlocking time of the converter
            tau: sampling time
            dead_time: specifies if a dead time of one sampling interval should be considered
        """
        self._interlocking_time = max(0, interlocking_time)
        self._tau = tau                     # sampling time
        self._dead_time = dead_time
        self._state = np.array([0, 0, 0])   # states of the three half bridges
        self._u_out = np.array([0, 0, 0])   # normalised output voltages of the converter
        # matrix for possible state transitions from row to column and the return value is the save state that needs to
        # be passed to consider interlocking times
        self._switching_states = np.array([[0, 1, 2], [0, 1, 0], [0, 0, 2]])
        self._dead_time_action = 0  # actions applied after the dead time of one time step
        self._current_action = 0  # currently applied action
        self._time_last_call = 0  # time of the last call of convert

    def convert(self, action, i_in, t):
        """
        The three output voltages of the converter are determined considering dead time and interlocking time.

        Args:
            action: desired switching state from the agent/controller
            i_in: input current into the motor
            t: current time

        Returns:
            Normalised output voltage of the converter
        """

        # take action from last time step if a new period starts
        # Do not take a new action if the interlocking interval was in the last call

        if self._dead_time:  # Consider dead time
            if t > self._time_last_call + self._interlocking_time:
                self._current_action = self._dead_time_action
                self._dead_time_action = action
            current_action = self._current_action
            self._time_last_call = t
        else:  # do not consider dead time
            current_action = action

        # Extract the actions for each half-bridge from the summarized action from the controller/agent
        # map the input action to the desired half-bridge states
        actions = np.array([current_action // 4, (current_action // 2) % 2, current_action % 2])+1

        for i in range(len(actions)):  # for each phase (half-bridge)
            # Get the necessary state transitions from the switching table. If an interlocking state is needed it is
            # applied instead of the desired action.
            self._state[i] = self._switching_states[self._state[i]][actions[i]]
            if self._state[i] == 1:  # the lower transistor is conducting
                self._u_out[i] = -1
            elif self._state[i] == 2:  # the upper transistor is conducting
                self._u_out[i] = 1
            elif self._state[i] == 0:   # both transistors are blocking
                self._u_out[i] = (np.sign(i_in[i])+1)/2
            else:
                print("Wrong action")  # If an impossible action should be applied.
        return self._u_out / 2


class Cont2Level3PhaseConverter:
    """
    This class includes the continuous controller using dynamic averaging. It is assumed that the PWM and B6 bridge
    works well, the desired output voltage is achieved and the interlocking error is added. The errors due to zero
    crossings of the current are neglected.
    A dead time of one time step is considered.
    The action space is continuous and contains normalised voltages between zero and one.
    """

    action_space = spaces.Box(-1, 1, shape=(3,))
    # Only positive voltages can be applied
    voltages = (-1, 1)
    # Positive and negative currents are possible
    currents = (-1, 1)

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
        interlocking time > 0 in seconds
        """
        self._interlocking_time = max(0, t)

    def __init__(self, interlocking_time=0, tau=1e-4, dead_time=True):
        """
        Basic setting of all the common parameter.

        Args:
            interlocking_time: interlocking time of the converter
            tau: sampling time
            dead_time: specifies if dead time of one sampling interval should be considered
        """
        self._interlocking_time = max(0, interlocking_time)
        self._tau = tau  # sampling time
        self._dead_time = dead_time
        self._dead_time_action = np.array([0, 0, 0])
        self._u_out = np.array([0, 0, 0])   # normalised output voltages of the converter

    def convert(self, action, i_in, *_):
        """
        The three output voltages of the converter are determined. The voltages are the actions itself and the
        interlocking error

         :math:`\\Delta U=-\\text{sign}(i_{in}) f_s {\\tau}_{dead time} U_{sup}`

        . Also a dead time of one time step
        is considered.

        Args:
            action: desired switching state from the agent/controller
            i_in: input current into the motor

        Returns:
            Normalised output voltage of the converter
        """

        if self._dead_time:  # Consider dead time
            current_action = self._dead_time_action  # Take action from last step because of dead time of one time step.
            self._dead_time_action = action
        else:  # do not consider dead time
            current_action = action

        self._u_out = current_action-np.sign(i_in)/self._tau*self._interlocking_time
        return self._u_out / 2

