import numpy as np

class MPCCurrentController:
    def __init__(self, env, env_id, prediction_horizon=1):
        self.env_id = env_id

        # Get attributes from environment
        self.state_names = env.get_wrapper_attr('state_names')
        self.physical_system = env.get_wrapper_attr('physical_system')
        self.tau = self.physical_system.tau
        self.limits = self.physical_system.limits

        motor = self.physical_system.electrical_motor.motor_parameter
        self.l_q = motor['l_q']
        self.l_d = motor['l_d']
        self.psi_ = motor['psi_p']
        self.r_s = motor['r_s']
        self.p = motor['p']

        # Indices
        self.i_sd_idx = self.state_names.index('i_sd')
        self.i_sq_idx = self.state_names.index('i_sq')
        self.omega_idx = self.state_names.index('omega')
        self.epsilon_idx = self.state_names.index('epsilon')

        self.u_sd_idx = self.state_names.index('u_sd')
        self.u_lim = self.limits[self.u_sd_idx]

        self.abc_to_dq = self.physical_system.abc_to_dq_space

        # Get finite set of voltage vectors (actions)
        self.subactions = -np.power(-1, self.physical_system._converter._subactions)
        self.u_abc_k1  = self.u_lim * self.subactions

        # Prediction horizon
        self.prediction_horizon = prediction_horizon
        print("MPCCurrentController initialized")


    def _simulate_sequence(self, i_d, i_q, epsilon_el, omega, ref_i_d, ref_i_q, depth):
        min_cost = float('inf')
        best_sequence = []

        for idx, (v_a, v_b, v_c) in enumerate(self.u_abc_k1):
            v_dq = np.transpose(
                np.array([self.abc_to_dq(np.array([v_a, v_b, v_c]), epsilon_el + 0.5 * omega * self.tau)])
            )
            v_d, v_q = v_dq[0], v_dq[1]

            epsilon_next = epsilon_el + omega * self.tau
            i_d_next = i_d + self.tau * ((v_d - self.r_s * i_d + omega * self.l_q * i_q) / self.l_d)
            i_q_next = i_q + self.tau * ((v_q - self.r_s * i_q - omega * self.l_d * i_d - omega * self.psi_) / self.l_q)

            cost = (i_d_next - ref_i_d)**2 + (i_q_next - ref_i_q)**2

            if depth == self.prediction_horizon - 1:
                total_cost = cost
                sequence = [idx]
            else:
                future_cost, future_sequence = self._simulate_sequence(
                    i_d_next, i_q_next, epsilon_next, omega, ref_i_d, ref_i_q, depth + 1
                )
                total_cost = cost + future_cost
                sequence = [idx] + future_sequence

            if total_cost < min_cost:
                min_cost = total_cost
                best_sequence = sequence

        return min_cost, best_sequence

    def control(self, state, reference):
        print("MPC control called with state:", state)
        i_d = state[self.i_sd_idx] * self.limits[self.i_sd_idx]
        i_q = state[self.i_sq_idx] * self.limits[self.i_sq_idx]
        epsilon_el = state[self.epsilon_idx] * self.limits[self.epsilon_idx]
        omega = self.p * state[self.omega_idx] * self.limits[self.omega_idx]

        ref_i_q = reference[0] * self.limits[self.i_sq_idx]
        ref_i_d = reference[1] * self.limits[self.i_sd_idx]

        _, best_sequence = self._simulate_sequence(
            i_d, i_q, epsilon_el, omega, ref_i_d, ref_i_q, depth=0
        )

        best_idx = best_sequence[0] if best_sequence else 0
        return best_idx

    def reset(self):
        pass

    def tune(self, env, env_id, **kwargs):
        pass

    @property
    def stages(self):
        return []
