
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gym_electric_motor as gem
from gem_controllers.gem_controller import GemController

SWITCHING_TABLE = np.array([
    [-1, -1, -1],  # V0
    [+1, -1, -1],  # V1
    [+1, +1, -1],  # V2
    [-1, +1, -1],  # V3
    [-1, +1, +1],  # V4
    [-1, -1, +1],  # V5
    [+1, -1, +1],  # V6
    [+1, +1, +1],  # V7
], dtype=int)

SUBACTIONS = np.where(SWITCHING_TABLE == 1, 1, -1)  
print("SUBACTIONS MATCH:")
print(SUBACTIONS)

class DTC_PMSM_Controller(GemController):
    def __init__(self, env, env_id, torque_hyst=2.0, flux_hyst=0.005):
        super().__init__()
        ps = env.unwrapped.physical_system
        mp = ps.electrical_motor.motor_parameter
        self.Ls = float(mp.get('l_s', 0.003))
        self.psi_p = float(mp.get('psi_p', 0.01))

        self.i_sd_idx = env.state_names.index('i_sd')
        self.i_sq_idx = env.state_names.index('i_sq')
        self.epsilon_idx = env.state_names.index('epsilon')
        self.torque_idx = env.state_names.index('torque')
        self.limits = ps.limits

        self.torque_hyst = torque_hyst
        self.flux_hyst = flux_hyst
        self.psi_ref = abs(self.psi_p)

        self.prev_dT = self.prev_dPsi = 0
        self.prev_sector = 0

        # 4. LOOKUP TABLE: 4 rows × 6 sectors = 24 entries
        self.table = np.array([
            [1, 5, 4, 6, 2, 3],   # dT=0, dΨ=0
            [5, 4, 6, 2, 3, 1],   # dT=0, dΨ=1
            [2, 3, 1, 5, 4, 6],   # dT=1, dΨ=0
            [6, 2, 3, 1, 5, 4],   # dT=1, dΨ=1
        ], dtype=int)

    def _flux(self, i_sd, i_sq, eps):
        psi_d = self.Ls * i_sd + self.psi_p * np.cos(eps)
        psi_q = self.Ls * i_sq + self.psi_p * np.sin(eps)
        return np.sqrt(psi_d**2 + psi_q**2), np.arctan2(psi_q, psi_d)


    def _sector(self, ang_rad):
        """
        Convert an angle in radians to one of six 60° sectors.
        Sector S1 is centered at 0° (covers -30° to +30°).

        Parameters:
            ang_rad (float): Angle in radians.

        Returns:
            int: Sector index (0 to 5), corresponding to:
                0 → -30° to +30°,
                1 → +30° to +90°,
                2 → +90° to +150°,
                3 → +150° to +210°,
                4 → +210° to +270°,
                5 → +270° to +330°.
        """
        # Convert radians → degrees, shift by 30°, wrap around 360°, divide into 6 sectors
        return int((np.degrees(ang_rad) + 30) % 360 // 60)


    def control(self, state, reference):
        i_sd = state[self.i_sd_idx] * self.limits[self.i_sd_idx]
        i_sq = state[self.i_sq_idx] * self.limits[self.i_sq_idx]
        eps = state[self.epsilon_idx] * 2 * np.pi
        T = state[self.torque_idx] * self.limits[self.torque_idx]
        T_ref = reference[0] * self.limits[self.torque_idx]

        psi, psi_ang = self._flux(i_sd, i_sq, eps)
        dT = T_ref - T
        dPsi = self.psi_ref - psi

        dT_bit = 1 if dT > self.torque_hyst else (0 if dT < -self.torque_hyst else self.prev_dT)
        dPsi_bit = 1 if dPsi > self.flux_hyst else (0 if dPsi < -self.flux_hyst else self.prev_dPsi)

        sector = self._sector(psi_ang)
        row = 2 * dT_bit + dPsi_bit
        idx = self.table[row, sector]

        if idx != self.prev_idx or sector != self.prev_sector:
            print(f"V{idx} | T_err={dT:+.3f} Nm | Ψ_err={dPsi:+.3f} Wb | S{sector+1}")

        self.prev_dT, self.prev_dPsi = dT_bit, dPsi_bit
        self.prev_idx = idx
        self.prev_sector = sector
        return idx  # ← returns [s_a, s_b, s_c]

    def reset(self):
        self.prev_dT = self.prev_dPsi = 0
        self.prev_idx = 7


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