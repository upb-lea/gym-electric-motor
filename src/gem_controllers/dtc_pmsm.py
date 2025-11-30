import numpy as np
from .gem_controller import GemController


class DTC_PMSM_Controller(GemController):
    """
    Classical Direct Torque Control (DTC) for PMSM with lookup table (Fig. 6-4).
    Uses:
      - Measured torque from environment
      - Stator flux estimated from currents
      - 6-sector division of the flux vector
      - 2-level hysteresis for torque and flux
      - Switching table → voltage vector index
    """
    VECTOR_TO_ACTION = {
    0: 0,
    1: 4,
    2: 6,
    3: 2,
    4: 3,
    5: 1,
    6: 5,
    7: 7,
    }

    def __init__(self, env, env_id, torque_hyst=0.01, flux_hyst=0.01):
        super().__init__()
        self.env_id = env_id
        ps_wrapper = env.unwrapped.physical_system
        self.step = getattr(ps_wrapper, 'dead_time', 0)

        # Environment & motor
        self.state_names = env.get_wrapper_attr('state_names')
        self.physical_system = env.get_wrapper_attr('physical_system')
        self.tau = self.physical_system.tau
        self.limits = self.physical_system.limits
        self.motor_params = self.physical_system.electrical_motor.motor_parameter
        for k, v in self.motor_params.items():
            setattr(self, k, v)

        self.pole_pairs = getattr(self, 'p', getattr(self, 'pole_pairs', 1))
        self.i_sd_idx = self.state_names.index('i_sd')
        self.i_sq_idx = self.state_names.index('i_sq')
        self.omega_idx = self.state_names.index('omega')
        self.torque_idx = self.state_names.index('torque')
        self.u_sd_idx = self.state_names.index('u_sd')
        self.u_lim = self.limits[self.u_sd_idx]

        # Discrete voltage vectors: 8 vectors (6 active + 2 zero)
        self.abc_to_dq = self.physical_system.abc_to_dq_space
        self.subactions = -np.power(-1, self.physical_system._converter._subactions)
        self.u_abc_k1 = self.u_lim * self.subactions  # shape (8, 3)

        # Hysteresis
        self.torque_hysteresis = torque_hyst
        self.flux_hysteresis = flux_hyst

        # Flux reference (usually |ψ_p|)
        # necessary motor parameter
        
        # Get stator inductance
        #self.L_s = self.motor_params.get('L_s')

        # Default L_d and L_q to L_s if they are missing
        self.Ld = self.motor_params.get('l_d')
        self.Lq = self.motor_params.get('l_q')
            # Previous vector (for zero-vector fallback)
        self.previous_voltage_idx = 0

        # Previous hysteresis states for holding
        self.prev_dT = 0
        self.prev_dΨ = 0

        self.dtc_table = np.array([
                [2, 3, 4, 5, 6, 1],  # Tdot>0, psidot>0
                [3, 4, 5, 6, 1, 2],  # Tdot>0, psidot<0
                [6, 1, 2, 3, 4, 5],  # Tdot<0, psidot>0
                [5, 6, 1, 2, 3, 4],  # Tdot<0, psidot<0
        ], dtype=int)

        # normal switching table, if the voltage vectors in the "traditional" order
        #self.dtc_table = np.array([
        #        [2, 3, 4, 5, 6, 1],  # Tdot>0, psidot>0
        #        [3, 4, 5, 6, 1, 2],  # Tdot>0, psidot<0
        #        [6, 1, 2, 3, 4, 5],  # Tdot<0, psidot>0
        #        [5, 6, 1, 2, 3, 4],  # Tdot<0, psidot<0
        #], dtype=int)

        # Zero vectors for when inside both bands
        self.zero_vectors = [0]  # V0 and V7; use V0 by default
        print("DTC pmsm")
       

    def _compute_psi_s_star(self, T_star):
        """
        Compute ψ_s* according to:
        ψ_s* = sqrt( ψ_p^2 + ( 2 L_s T* / (3 p ψ_p) )^2 )
        where T* comes from the system state vector x using torque_idx.
        """
        

        # Motor parameters already stored earlier
        psi_p = self.psi_p
        L_q  = self.motor_params.get('l_q')
        p = self.motor_params.get('p')


        # Formula
        term = (2 * L_q * T_star) / (3 * p * psi_p)
        psi_s_star = np.sqrt(psi_p**2 + term**2)

        return psi_s_star


    def _estimate_flux(self, i_sd, i_sq, epsilon):
        """Estimate stator flux magnitude and angle from currents."""

        
        psi_d = self.Ld * i_sd + self.psi_p * np.cos(epsilon)
        psi_q = self.Lq * i_sq + self.psi_p * np.sin(epsilon)
        psi_mag = np.sqrt(psi_d**2 + psi_q**2)
        psi_angle = np.arctan2(psi_q, psi_d)
        return psi_mag, psi_angle

    def _get_sector(self, angle_rad):
        """Convert stator flux angle to sector 1..6 (centered at 0°,60°,...,300°)."""
        angle_deg = np.degrees(angle_rad) % 360
        sector = int((angle_deg + 30) // 60) + 1  # Offset by 30° for sector boundaries
        return min(sector, 6)  # Clamp to 1-6

    def control(self, state, reference):
        # --- 1. Read physical states (denormalize) ---
        i_sd = state[self.i_sd_idx] * self.limits[self.i_sd_idx]
        i_sq = state[self.i_sq_idx] * self.limits[self.i_sq_idx]
        epsilon_idx = self.state_names.index('epsilon')
        epsilon = state[epsilon_idx] * self.limits[epsilon_idx]  # Assuming epsilon normalized to [0,1] or [-pi,pi]; adjust if needed
        omega = state[self.omega_idx] * self.limits[self.omega_idx]
        torque_meas = state[self.torque_idx] * self.limits[self.torque_idx]
        #self.torque_ref = reference[0] * self.limits[self.torque_idx]
        torque_ref = reference[0] * self.limits[self.torque_idx]

        # --- 2. Estimate flux ---
        psi_ref = self._compute_psi_s_star(torque_ref)
        psi_mag, psi_angle = self._estimate_flux(i_sd, i_sq, epsilon)

        # --- 3. 2-LEVEL HYSTERESIS COMPARATORS ---
        T_err = torque_ref - torque_meas
        Ψ_err = psi_ref - psi_mag

        # Torque hysteresis: 1 (increase), 0 (decrease/no change)
        if T_err > self.torque_hysteresis:
            dT = 1
        elif T_err < -self.torque_hysteresis:
            dT = 0
        else:
            dT = self.prev_dT  # Hold previous state

        # Flux hysteresis: 1 (increase), 0 (decrease/no change)
        if Ψ_err > self.flux_hysteresis:
            dΨ = 1
        elif Ψ_err < -self.flux_hysteresis:
            dΨ = 0
        else:
            dΨ = self.prev_dΨ  # Hold previous state

        # --- 4. Zero vector if inside both hysteresis bands ---
        inside_T_band = abs(T_err) <= self.torque_hysteresis
        inside_Ψ_band = abs(Ψ_err) <= self.flux_hysteresis

        if inside_T_band and inside_Ψ_band:
            vector_idx = self.zero_vectors[0]
        # if dT == self.prev_dT and dΨ == self.prev_dΨ:
        #     vector_idx = self.zero_vectors[0]  # Default to V0
        else:
            # --- 5. Determine sector (1-based) ---
            sector = self._get_sector(psi_angle)

            # --- 6. Table lookup (row: 2*dT + (1-dΨ), col: sector-1) ---
            # row = 2 * dT + (1 - dΨ)
            row = (1 - dT) * 2 + (1 - dΨ)   # Maps to 0-3: dT=1 dΨ=1 →0, dT=1 dΨ=0 →1, etc.
            vector_idx = self.dtc_table[row, sector - 1]
            #print(row, sector)

        action = self.VECTOR_TO_ACTION[vector_idx]
        
        # --- 7. Optional: Dead-time compensation (simple hold if step>0) ---
        if self.step > 0:
            action = self.previous_voltage_idx

        # --- 8. Commit and return ---
        self.prev_dT = dT
        self.prev_dΨ = dΨ
        self.previous_voltage_idx = action
        return int(action)

    def reset(self):
        """Reset controller state."""
        self.previous_voltage_idx = 7  # Start with V0
        self.prev_dT = 0
        self.prev_dΨ = 0

    def make(env, env_id, **kwargs):  
        print(f"[DTC] Building for {env_id}")

        torque_hyst = kwargs.pop('torque_hyst', 0.05)  # default 5%
        flux_hyst = kwargs.pop('flux_hyst', 0.02)

        controller = DTC_PMSM_Controller(
            env, env_id,
            torque_hyst=torque_hyst,
            flux_hyst=flux_hyst
        )

        if kwargs.pop('_skip_outer_loops', False):
            print("[DTC] Standalone mode")
            controller._stages = []
            return controller

        return controller