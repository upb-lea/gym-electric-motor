import numpy as np

from .electric_motor import ElectricMotor


class DcMotor(ElectricMotor):
    """
        The DcMotor and its subclasses implement the technical system of a dc motor.

        This includes the system equations, the motor parameters of the equivalent circuit diagram,
        as well as limits.

        =====================  ==========  ============= ===========================================
        Motor Parameter        Unit        Default Value Description
        =====================  ==========  ============= ===========================================
        r_a                    Ohm         0.78          Armature circuit resistance
        r_e                    Ohm         25            Exciting circuit resistance
        l_a                    H           6.3e-3        Armature circuit inductance
        l_e                    H           1.2           Exciting circuit inductance
        l_e_prime              H           0.0094        Effective excitation inductance
        j_rotor                kg/m^2      0.017         Moment of inertia of the rotor
        =====================  ==========  ============= ===========================================

        =============== ====== =============================================
        Motor Currents  Unit   Description
        =============== ====== =============================================
        i_a             A      Armature circuit current
        i_e             A      Exciting circuit current
        =============== ====== =============================================
        =============== ====== =============================================
        Motor Voltages  Unit   Description
        =============== ====== =============================================
        u_a             V      Armature circuit voltage
        u_e             v      Exciting circuit voltage
        =============== ====== =============================================

        ======== ===========================================================
        Limits / Nominal Value Dictionary Entries:
        -------- -----------------------------------------------------------
        Entry    Description
        ======== ===========================================================
        i_a      Armature current
        i_e      Exciting current
        omega    Angular Velocity
        torque   Motor generated torque
        u_a      Armature Voltage
        u_e      Exciting Voltage
        ======== ===========================================================
    """

    # Indices for array accesses
    I_A_IDX = 0
    I_E_IDX = 1
    CURRENTS_IDX = [0, 1]
    CURRENTS = ['i_a', 'i_e']
    VOLTAGES = ['u_a', 'u_e']

    # Motor parameter, nominal values and limits are based on the following DC Motor:
    # https://www.heinzmann-electric-motors.com/en/products/dc-motors/pmg-132-dc-motor
    _default_motor_parameter = {
        'r_a': 16e-3, 'r_e': 16e-2, 'l_a': 19e-6, 'l_e_prime': 1.7e-3, 'l_e': 5.4e-3, 'j_rotor': 0.0025
    }
    _default_nominal_values = dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
    _default_limits = dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)
    _default_initializer = {'states': {'i_a': 0.0, 'i_e': 0.0},
                            'interval': None,
                            'random_init': None,
                            'random_params': (None, None)}

    def __init__(self, motor_parameter=None, nominal_values=None, limit_values=None, motor_initializer=None):
        # Docstring of superclass
        super().__init__(motor_parameter, nominal_values, limit_values, motor_initializer)
        #: Matrix that contains the constant parameters of the systems equation for faster computation
        self._model_constants = None
        self._update_model()
        self._update_limits()

    def _update_model(self):
        """Updates the motor's model parameters with the motor parameters.

        Called internally when the motor parameters are changed or the motor is initialized.
        """
        mp = self._motor_parameter
        self._model_constants = np.array(
            [
                [-mp['r_a'], 0, -mp['l_e_prime'], 1, 0],
                [0, -mp['r_e'], 0, 0, 1]
            ]
        )
        self._model_constants[self.I_A_IDX] = self._model_constants[self.I_A_IDX] / mp['l_a']
        self._model_constants[self.I_E_IDX] = self._model_constants[self.I_E_IDX] / mp['l_e']

    def torque(self, currents):
        # Docstring of superclass
        return self._motor_parameter['l_e_prime'] * currents[self.I_A_IDX] * currents[self.I_E_IDX]

    def i_in(self, currents):
        # Docstring of superclass
        return list(currents)

    def electrical_ode(self, state, u_in, omega, *_):
        # Docstring of superclass
        return np.matmul(
            self._model_constants,
            np.array([
                state[self.I_A_IDX],
                state[self.I_E_IDX],
                omega * state[self.I_E_IDX],
                u_in[0],
                u_in[1],
            ])
        )

    def get_state_space(self, input_currents, input_voltages):
        """
        Calculate the possible normalized state space for the motor as a tuple of dictionaries "low" and "high".

        Args:
            input_currents: Tuple of the two converters possible output currents.
            input_voltages: Tuple of the two converters possible output voltages.

        Returns:
             tuple(dict,dict): Dictionaries defining if positive and negative values are possible for each motors state.
        """
        a_converter = 0
        e_converter = 1
        low = {
            'omega': -1 if input_voltages.low[a_converter] == -1
                           or input_voltages.low[e_converter] == -1 else 0,
            'torque': -1 if input_currents.low[a_converter] == -1
                            or input_currents.low[e_converter] == -1 else 0,
            'i_a': -1 if input_currents.low[a_converter] == -1 else 0,
            'i_e': -1 if input_currents.low[e_converter] == -1 else 0,
            'u_a': -1 if input_voltages.low[a_converter] == -1 else 0,
            'u_e': -1 if input_voltages.low[e_converter] == -1 else 0,
        }
        high = {
            'omega': 1,
            'torque': 1,
            'i_a': 1,
            'i_e': 1,
            'u_a': 1,
            'u_e': 1
        }
        return low, high

    def _update_limits(self, limits_d=None, nominal_d=None):
        # Docstring of superclass
        if limits_d is None:
            limits_d = dict()

        # torque is replaced the same way for all DC motors
        limits_d.update(dict(torque=self.torque([self._limits[state] for state in self.CURRENTS])))
        super()._update_limits(limits_d)
