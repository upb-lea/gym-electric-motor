import numpy as np
from .gem_controller import GemController

class MPCCurrentController(GemController):
    """
    Finite-Control Set Model Predictive Current Controller for PMSM and SynRM motors.
    
    This controller implements FCS-MPC to directly evaluate switching states
    and optimize current tracking in field-oriented coordinates.
    
    Args:
        env: Gym environment instance
        env_id: Environment ID string
        prediction_horizon: Prediction horizon length (default: 1)
        w_d: Weight for d-axis current error (default: 1.0)
        w_q: Weight for q-axis current error (default: 1.0)
    """
    
    def __init__(self, env, env_id, prediction_horizon=1, w_d=1.0, w_q=1.0):
        super().__init__()
        self.env_id = env_id
        self.prediction_horizon = prediction_horizon
        self.w_d = w_d
        self.w_q = w_q        

        # Assign self.step from the environment wrapper, default to 0 if no DeadTimeProcessor
        ps_wrapper = env.unwrapped.physical_system
        self.step = getattr(ps_wrapper, 'dead_time', 0)
        print(f"DeadTimeProcessor steps: {self.step}")   

        # Retrieve environment info and motor parameters
        self.state_names = env.get_wrapper_attr('state_names')
        self.physical_system = env.get_wrapper_attr('physical_system')
        self.tau = self.physical_system.tau
        self.limits = self.physical_system.limits
        self.motor_params = self.physical_system.electrical_motor.motor_parameter
        for key, value in self.motor_params.items():
            setattr(self, key, value)

        # Identify indices of key states and inputs
        self.i_sd_idx = self.state_names.index('i_sd')
        self.i_sq_idx = self.state_names.index('i_sq') 
        self.omega_idx = self.state_names.index('omega')        
        self.u_sd_idx = self.state_names.index('u_sd')
        self.u_lim = self.limits[self.u_sd_idx]

        # Setup coordinate transforms and precompute voltage combinations
        self.abc_to_dq = self.physical_system.abc_to_dq_space
        self.subactions = -np.power(-1, self.physical_system._converter._subactions)
        self.u_abc_k1 = self.u_lim * self.subactions

        # Load motor model constants and motor-specific state names
        self._model_constants = self.physical_system.electrical_motor._model_constants
        motor_type = type(self.physical_system.electrical_motor).__name__
        if motor_type in ["PermanentMagnetSynchronousMotor", "SynchronousReluctanceMotor"]:
            self.motor_state_names = ["i_sd", "i_sq", "epsilon"]
        else:
            raise NotImplementedError(f"MPC controller not implemented for motor type: {motor_type}")

        # Initialize delay compensation variables
        self.previous_voltage_idx = 0
        self.past_references = []
        self.extrapolation_order = 2

    def _extrapolate_reference(self, current_ref, n=2):
        """
        [REFERENCE IMPLEMENTATION - Not currently active]
        
        Extrapolate reference for delay compensation using past references. 
        Works for sinusoidal signals in alpha-beta frames.
        
        This method uses polynomial extrapolation to predict future reference
        values based on past samples. Useful for systems with computational delays.
        
        Args:
            current_ref: Current reference value
            n: Extrapolation order (default: 2 for quadratic extrapolation)
            
        Returns:
            extrapolated_ref: Extrapolated reference value
        """
        # Implementation kept as reference for future use
        # Uncomment and modify as needed for specific applications
        """
        self.past_references.append(current_ref.copy())
        if len(self.past_references) > n + 1:
            self.past_references.pop(0)
        if len(self.past_references) < n + 1:
            return current_ref
        ref_k = self.past_references[-1]
        ref_km1 = self.past_references[-2]
        ref_km2 = self.past_references[-3]
        return 6 * ref_k - 8 * ref_km1 + 3 * ref_km2
        """
        # Currently using state estimation instead of reference extrapolation
        return current_ref

    def _estimate_currents(self, model_constants, x, omega, voltage_idx):
        """
        Estimate currents at next timestep using previous voltage for delay compensation.
        
        Args:
            model_constants: Motor model constants matrix
            x: Current state vector [i_sd, i_sq, epsilon]
            omega: Electrical rotor speed
            voltage_idx: Index of previously applied voltage vector
            
        Returns:
            Estimated state vector at next timestep
        """
        v_abc = self.u_abc_k1[voltage_idx]
        v_dq = np.transpose(
            np.array([self.abc_to_dq(v_abc, x[-1] + 0.5 * omega * self.tau)])
        )
        v_d, v_q = v_dq[0], v_dq[1]
        omega_Isd = omega * x[0]
        omega_Isq = omega * x[1]
        ext_vec = np.array([omega, x[0], x[1], float(v_d), float(v_q), omega_Isd, omega_Isq])
        dx = model_constants @ ext_vec
        return x + self.tau * dx

    def _simulate_sequence(self, model_constants, x, omega, ref_i_d, ref_i_q, depth):
        """
        Simulate all possible voltage sequences to find the one minimizing the cost.
        
        Args:
            model_constants: Motor model constants matrix
            x: Current state vector
            omega: Electrical rotor speed
            ref_i_d: D-axis current reference
            ref_i_q: Q-axis current reference
            depth: Current depth in prediction horizon
            
        Returns:
            min_cost: Minimum cost found
            best_sequence: Best voltage sequence
        """
        min_cost = float('inf')
        best_sequence = []

        for idx, (v_a, v_b, v_c) in enumerate(self.u_abc_k1):
            v_dq = np.transpose(
                np.array([self.abc_to_dq(np.array([v_a, v_b, v_c]), x[-1] + 0.5 * omega * self.tau)])
            )
            v_d, v_q = v_dq[0], v_dq[1]
            omega_Isd = omega * x[0]
            omega_Isq = omega * x[1]
            ext_vec = np.array([omega, x[0], x[1], float(v_d), float(v_q), omega_Isd, omega_Isq])
            dx = model_constants @ ext_vec
            x_next = x + self.tau * dx

            # Compute cost based on tracking error
            cost = self.w_d * (x_next[0] - ref_i_d) ** 2 + self.w_q * (x_next[1] - ref_i_q) ** 2

            if depth == self.prediction_horizon - 1:
                total_cost, sequence = cost, [idx]
            else:
                future_cost, future_sequence = self._simulate_sequence(
                    model_constants, x_next, omega, ref_i_d, ref_i_q, depth + 1
                )
                total_cost = cost + future_cost
                sequence = [idx] + future_sequence

            if total_cost < min_cost:
                min_cost = total_cost
                best_sequence = sequence

        return min_cost, best_sequence

    def control(self, state, reference):
        """
        Compute the optimal voltage vector based on current state and reference.
        
        Args:
            state: Current motor state vector
            reference: Current reference vector [i_sd_ref, i_sq_ref]
            
        Returns:
            best_idx: Index of optimal voltage vector
        """
        x_measured = np.array([state[self.state_names.index(n)] * self.limits[self.state_names.index(n)]
                               for n in self.motor_state_names])
        omega = state[self.omega_idx] * self.limits[self.omega_idx]        
        ref_i_d = reference[0] * self.limits[self.i_sd_idx]
        ref_i_q = reference[1] * self.limits[self.i_sq_idx]

        if self.step == 0:
            # Without delay compensation
            _, best_sequence = self._simulate_sequence(
                self._model_constants, x_measured, omega, ref_i_d, ref_i_q, depth=0
            )
        else:
            # With delay compensation
            x_est = self._estimate_currents(self._model_constants, x_measured, omega, self.previous_voltage_idx)            
            _, best_sequence = self._simulate_sequence(
                self._model_constants, x_est, omega, ref_i_d, ref_i_q, depth=0
            )

        best_idx = best_sequence[0] if best_sequence else 0
        self.previous_voltage_idx = best_idx
        return best_idx

    def reset(self):
        """
        Reset the controller state including delay compensation variables.
        """
        self.previous_voltage_idx = 0
        self.past_references = []