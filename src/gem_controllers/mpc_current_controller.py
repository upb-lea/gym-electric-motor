import numpy as np
from .gem_controller import GemController

class MPCCurrentController(GemController):
    """MPC current controller with delay compensation using extrapolation"""
    
    @property
    def signal_names(self):
        return ["u_MPC"]
    
    @property
    def stages(self):
        return []
    
    @property
    def references(self):
        return dict()
    
    @property
    def referenced_states(self):
        return ['i_sd', 'i_sq']
    
    @property
    def maximum_reference(self):
        return {'i_sd': 1.0, 'i_sq': 1.0}
    
    def __init__(self, env, env_id, prediction_horizon=1):
        super().__init__()
        self.env_id = env_id
        
        # Get attributes from environment
        self.state_names = env.get_wrapper_attr('state_names')
        self.physical_system = env.get_wrapper_attr('physical_system')
        self.tau = self.physical_system.tau
        self.limits = self.physical_system.limits

        self.motor_params = self.physical_system.electrical_motor.motor_parameter

        # Store each parameter as an attribute dynamically
        for key, value in self.motor_params.items():
            setattr(self, key, value)      
        
        # State indices
        self.i_sd_idx = self.state_names.index('i_sd')
        self.i_sq_idx = self.state_names.index('i_sq') 
        self.omega_idx = self.state_names.index('omega')        
        self.u_sd_idx = self.state_names.index('u_sd')
        self.u_lim = self.limits[self.u_sd_idx]

        # Coordinate transformation
        self.abc_to_dq = self.physical_system.abc_to_dq_space
        self.subactions = -np.power(-1, self.physical_system._converter._subactions)
        
        # All possible voltage vectors in abc coordinates  
        self.u_abc_k1 = self.u_lim * self.subactions
        
        self.prediction_horizon = prediction_horizon
        
        # Model constants for MPC
        self._model_constants = self.physical_system.electrical_motor._model_constants  
        motor_type = type(self.physical_system.electrical_motor).__name__

        if motor_type == "PermanentMagnetSynchronousMotor":
            self.motor_state_names = ["i_sd", "i_sq", "epsilon"]
        elif motor_type == "SynchronousReluctanceMotor":
            self.motor_state_names = ["i_sd", "i_sq", "epsilon"]        
        else:
            raise NotImplementedError(f"MPC controller not implemented for motor type: {motor_type}")        
          
        # === DELAY COMPENSATION VARIABLES ===
        self.previous_voltage_idx = 0  # Store the previously calculated voltage
        self.past_references = []  # Store past references for extrapolation
        self.extrapolation_order = 2  # n=2 as recommended in the text
        
        print("MPCCurrentController with delay compensation initialized")

    def _extrapolate_reference(self, current_ref, n=2):
        """Extrapolate future reference using Lagrange extrapolation (Eq. 12.8-12.10)"""
        # Store current reference
        self.past_references.append(current_ref.copy())
        
        # Keep only the needed history (n+1 samples)
        if len(self.past_references) > n + 1:
            self.past_references.pop(0)
        
        # If we don't have enough history, return current reference
        if len(self.past_references) < n + 1:
            return current_ref
        
        # Extract references: i*(k), i*(k-1), i*(k-2), etc.
        ref_k = self.past_references[-1]      # i*(k)
        ref_k_minus_1 = self.past_references[-2]  # i*(k-1)
        ref_k_minus_2 = self.past_references[-3]  # i*(k-2)      
  
        # Extrapolate i*(k+2) 
        ref_k_plus_2 = 6 * ref_k - 8 * ref_k_minus_1 + 3 * ref_k_minus_2
        
        return  ref_k_plus_2

    def _estimate_currents(self, model_constants, x, omega, voltage_idx):
        """Estimate currents at next sampling instant (Step 3 in flowchart)"""
        # Get the voltage vector that was applied in the previous cycle
        v_abc = self.u_abc_k1[voltage_idx]
        
        # Transform voltages to dq frame at the appropriate angle
        v_dq = np.transpose(
            np.array([self.abc_to_dq(v_abc, x[-1] + 0.5 * omega * self.tau)])
        )
        v_d, v_q = v_dq[0], v_dq[1]
        
        # Calculate omega * i terms
        omega_Isd = omega * x[0]
        omega_Isq = omega * x[1]
        
        # Build input vector for derivative calculation
        exterded_vector = np.array([
            omega,
            x[0],       # i_d
            x[1],       # i_q
            float(v_d), # u_d
            float(v_q), # u_q
            omega_Isd,
            omega_Isq
        ])
        
        # Compute derivative using the applied voltage
        dx = model_constants @ exterded_vector  
        
        # Euler integration to estimate next state
        x_estimated = x + self.tau * dx
        
        return x_estimated

    def _simulate_sequence(self, model_constants, x_estimated, omega, ref_i_d, ref_i_q, depth):
        """Predict future states from estimated current (Step 4 in flowchart)"""
        min_cost = float('inf')
        best_sequence = []

        for idx, (v_a, v_b, v_c) in enumerate(self.u_abc_k1):
            # Transform voltages to dq frame
            v_dq = np.transpose(
                np.array([self.abc_to_dq(np.array([v_a, v_b, v_c]), x_estimated[-1] + 0.5 * omega * self.tau)])
            )
            v_d, v_q = v_dq[0], v_dq[1]

            # Calculate omega * i terms for estimated state
            omega_Isd = omega * x_estimated[0]
            omega_Isq = omega * x_estimated[1]

            # Build input vector: [omega, i_d, i_q, u_d, u_q, omega*i_d, omega*i_q]
            exterded_vector = np.array([
                omega,
                x_estimated[0],  # i_d (estimated)
                x_estimated[1],  # i_q (estimated)
                float(v_d),      # u_d
                float(v_q),      # u_q
                omega_Isd,
                omega_Isq
            ])

            # Compute derivative
            dx = model_constants @ exterded_vector  
            
            # Euler integration for next step (predicting k+2 from estimated k+1)
            x_next = x_estimated + self.tau * dx

            # Calculate cost using future reference (k+2)
            cost = (x_next[0] - ref_i_d) ** 2 + (x_next[1] - ref_i_q) ** 2

            if depth == self.prediction_horizon - 1:
                total_cost = cost
                sequence = [idx]
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
        """Main control method with delay compensation"""
        # Step 1: Measurement (k)
        state_vector = []
        for name in self.motor_state_names:
            s_idx = self.state_names.index(name)
            state_vector.append(state[s_idx] * self.limits[s_idx])
        x_measured = np.array(state_vector)  

        # Denormalize state
        omega = state[self.omega_idx] * self.limits[self.omega_idx]           
        
        # Step 2: Apply previously calculated voltage (k)
        # (This happens automatically in the environment - we just track what was applied)
        
        # Step 3: Estimate currents at k+1
        x_estimated = self._estimate_currents(
            self._model_constants, x_measured, omega, self.previous_voltage_idx
        )
        
        # Step 4: Extrapolate future references
        ref_i_q_current = reference[0] * self.limits[self.i_sq_idx]
        ref_i_d_current = reference[1] * self.limits[self.i_sq_idx]
        current_ref = np.array([ref_i_d_current, ref_i_q_current])
        
        # Get references for k+2
        ref_k_plus_2 = self._extrapolate_reference(current_ref)
        ref_i_d_future, ref_i_q_future = ref_k_plus_2  # Use k+2 reference for prediction
        
        # Step 5: Predict for k+2 and evaluate cost function
        _, best_sequence = self._simulate_sequence(
            self._model_constants, x_estimated, omega, 
            ref_i_d_future, ref_i_q_future, depth=0
        )
        
        # Step 6: Select optimal switching state
        best_idx = best_sequence[0] if best_sequence else 0
        self.previous_voltage_idx = best_idx  # Store for next cycle
        
        return best_idx

    def tune(self, env, env_id, **kwargs):
        """Required tuning method"""
        pass

    def reset(self):
        """Reset controller state including delay compensation variables"""
        self.previous_voltage_idx = 0
        self.past_references = []