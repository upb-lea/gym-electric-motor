import numpy as np
from .gem_controller import GemController


class MPCCurrentController(GemController):
    """MPC current controller for finite action space control"""
    
    @property
    def signal_names(self):
        """Signal names of the calculated values"""
        return ["u_MPC"]
    
    @property
    def stages(self):
        """List of stages (empty for MPC as it handles everything internally)"""
        return []
    
    @property
    def references(self):
        """Reference values of the current control stage"""
        return dict()
    
    @property
    def referenced_states(self):
        """Referenced states of the current control stage"""
        return ['i_sd', 'i_sq']
    
    @property
    def maximum_reference(self):
        """Maximum reference values"""
        return {'i_sd': 1.0, 'i_sq': 1.0}
    
    def __init__(self, env, env_id, prediction_horizon=1):
        """
        Initialize MPC current controller
        
        Args:
            env: GEM environment
            env_id: Environment ID
            prediction_horizon: MPC prediction steps
        """
        super().__init__()
        self.env_id = env_id
        
        # Get attributes from environment
        self.state_names = env.get_wrapper_attr('state_names')
        self.physical_system = env.get_wrapper_attr('physical_system')
        self.tau = self.physical_system.tau
        self.limits = self.physical_system.limits

        #motor_type = self.physical_system.electrical_motor.motor_type
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
        
        print("MPCCurrentController initialized")

        # Model constants for MPC
        self._model_constants = self.physical_system.electrical_motor._model_constants  
        motor_type = type(self.physical_system.electrical_motor).__name__

        if motor_type == "PermanentMagnetSynchronousMotor":
            self.motor_state_names = ["i_sd", "i_sq", "epsilon"]
        elif motor_type == "SynchronousReluctanceMotor":
            self.motor_state_names = ["i_sd", "i_sq", "epsilon"]        
        else:
         raise NotImplementedError(f"MPC controller not implemented for motor type: {motor_type}")        
          
        

    def _simulate_sequence(self,model_constants,x,omega,omega_Isd,omega_Isq, ref_i_d, ref_i_q, depth):
        min_cost = float('inf')
        best_sequence = []

        for idx, (v_a, v_b, v_c) in enumerate(self.u_abc_k1):
            # Transform voltages to dq frame
            v_dq = np.transpose(
                np.array([self.abc_to_dq(np.array([v_a, v_b, v_c]), x[-1] + 0.5 * omega * self.tau)])
            )
            v_d, v_q = v_dq[0], v_dq[1]

            # Build input vector: [omega, i_d, i_q, u_d, u_q, omega*i_d, omega*i_q]
            exterded_vector = np.array([
                omega,
                x[0],       # i_d
                x[1],       # i_q
                float(v_d),      # u_d
                float(v_q),      # u_q
                omega_Isd,
                omega_Isq
            ])

            # Compute derivative
            dx = model_constants @ exterded_vector  
            
            # Euler integration for next step
            x_next = x + self.tau * dx

            # Extract next states
            i_d_next, i_q_next, epsilon_next = x_next
            omega_Isd_next = omega * i_d_next
            omega_Isq_next = omega * i_q_next
            
            # Calculate cost
            cost = (i_d_next  - ref_i_d) ** 2 + (i_q_next - ref_i_q) ** 2

            if depth == self.prediction_horizon - 1:
                total_cost = cost
                sequence = [idx]
            else:
                future_cost, future_sequence = self._simulate_sequence(
                model_constants, x_next, omega,
                omega_Isd_next, omega_Isq_next,
                ref_i_d, ref_i_q, depth + 1
            )
                total_cost = cost + future_cost
                sequence = [idx] + future_sequence

            if total_cost < min_cost:
                min_cost = total_cost
                best_sequence = sequence

        return min_cost, best_sequence

    def control(self, state, reference):
        """Main control method matching PI controller interface"""      
    
        # Build denormalized state vector for the motor (only relevant states)
        state_vector = []
        for name in self.motor_state_names:
            s_idx = self.state_names.index(name)
            state_vector.append(state[s_idx] * self.limits[s_idx])
        x = np.array(state_vector)  

        # Denormalize state and references        
        omega = state[self.omega_idx] * self.limits[self.omega_idx]           
        ref_i_q = reference[0] * self.limits[self.i_sq_idx]
        ref_i_d = reference[1] * self.limits[self.i_sq_idx]  
        omega_Isd = omega * x[0]  # omega * i_sd
        omega_Isq = omega * x[1]  # omega * i_sq    

        # Unpack model constants
        model_constants= self.physical_system.electrical_motor._model_constants   
     
        # Pass everything needed to _simulate_sequence
        _, best_sequence = self._simulate_sequence(
            model_constants,x,omega,omega_Isd,omega_Isq, ref_i_d, ref_i_q, depth=0
        )
        best_idx = best_sequence[0] if best_sequence else 0
        return best_idx

    def tune(self, env, env_id, **kwargs):
        """Required tuning method (no tuning needed for MPC)"""
        pass

    def reset(self):
        """Reset controller state"""
        pass