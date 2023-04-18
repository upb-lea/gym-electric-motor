import random
import numpy as np
from numba import njit
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator, griddata
from pathlib import Path


# complex rotation element vectors as reference for clipping
ROTATION_MAP = np.ones((2, 2, 2), dtype=np.complex64)
ROTATION_MAP[1, 0, 1] = 0.5*(1+ np.sqrt(3)*1j)
ROTATION_MAP[1, 1, 0] = 0.5*(1- np.sqrt(3)*1j)
ROTATION_MAP[0, 1, 0] = 0.5*(-1- np.sqrt(3)*1j)
ROTATION_MAP[0, 1 ,1] = -1
ROTATION_MAP[0, 0, 1] = 0.5*(-1 + np.sqrt(3)*1j)



class FastPMSM:
    """ 'If everything seems under control, you're just not going fast enough.' - Mario Andretti"""

    t32 = np.array([[1, 0], [-0.5, 0.5 * np.sqrt(3)],  # only for alpha/beta -> abc
                   [-0.5, -0.5 * np.sqrt(3)]])
    t23 = 2/3 * np.array([[1, 0], [-0.5, 0.5 * np.sqrt(3)],  # only for abc -> alpha/beta
                         [-0.5, -0.5 * np.sqrt(3)]]).T
    tau = 1e-4
    i_limit = 400  # in A
    u_limit = 200  # u_DC/2 in V
    # compatibility with gym
    action_space = np.array([-1.0, 1.0])

    def __init__(self, x_star, batch_size=128, r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, me_omega=1000/60*2*np.pi,
                 saturated=False, debug=False):

        self.r_s_const = r_s
        self.me_omega_const = me_omega
        self.el_omega_const = self.me_omega_const*3
        self.l_d_const = l_d
        self.l_q_const = l_q
        self.psi_p_const = psi_p

        self.batch_size = batch_size  # automatically updating the batchdim of the constants

        self.x_star = x_star
        self.cntr = 0
        self.idx_sample = None
        self.ref = None
        self.k = None
        self.u_dq = None
        self.debug = debug

        # saturated model
        self.saturated = saturated
        if saturated:
            saturated_quants = ["L_dd_map", "L_dq_map",
                                "L_qd_map", "L_qq_map", "Psi_d_map", "Psi_q_map", ]
            self.pmsm_lut = loadmat(Path.cwd() / 'Reforce_PMSM_LUT.mat')
            for q in saturated_quants:
                qmap = self.pmsm_lut[q]
                x, y = np.indices(qmap.shape)
                nan_mask = np.isnan(qmap)
                qmap[nan_mask] = griddata(
                                (x[~nan_mask], y[~nan_mask]), # points we know
                                qmap[~nan_mask],    # values we know
                                (x[nan_mask], y[nan_mask]), # points to interpolate
                                method='nearest')   # extrapolation can only do nearest
                self.pmsm_lut[q] = qmap
            
            i_max = self.i_limit
            n_grid_points_y, n_grid_points_x = self.pmsm_lut[saturated_quants[0]].shape
            x, y = np.linspace(-i_max, 0, n_grid_points_x), \
                np.linspace(-i_max, i_max, n_grid_points_y)
            self.interpolators = {q: RegularGridInterpolator((x, y), self.pmsm_lut[q][:, :].T,
                                                             method='linear', bounds_error=False,
                                                             fill_value=None)
                                  for q in saturated_quants}


    def update_batch_dim(self):
        self.eps = np.zeros(self.batch_size)
        self.r_s = np.full((self.batch_size,), self.r_s_const)
        self.me_omega = np.full((self.batch_size,), self.me_omega_const)
        self.el_omega = np.full((self.batch_size,), self.me_omega_const*3)
        self.l_d = np.full((self.batch_size,), self.l_d_const)
        self.l_q = np.full((self.batch_size,), self.l_q_const)
        self.l_dq = np.column_stack([self.l_d_const, self.l_q_const])
        self.psi_p = np.full((self.batch_size,), self.psi_p_const)
        self.i_dq = np.zeros((self.batch_size, 2))

    @staticmethod
    def get_advanced_angle(eps, tau_scale, tau, omega):
        return eps + tau_scale * tau * omega

    @staticmethod
    def step_eps(eps):
        eps += 0.01*np.pi
        eps %= (2*np.pi)
        eps[eps > np.pi] -= 2*np.pi
        return eps
    
    @staticmethod
    @njit
    def t_dq_alpha_beta(eps):
        cos = np.cos(eps)
        sin = np.sin(eps)
        return np.column_stack((cos, sin, -sin, cos)).reshape(-1, 2, 2)

    def dq2abc(self, u_dq, eps):
        u_abc = self.t32 @ self.dq2albet(u_dq, eps).T
        return u_abc.T

    def dq2albet(self, u_dq, eps):
        q = self.t_dq_alpha_beta(-eps)
        u_alpha_beta = np.einsum("bij,bj->bi", q, u_dq)
        return u_alpha_beta

    def albet2dq(self, u_albet, eps):
        q_inv = self.t_dq_alpha_beta(eps)
        u_dq = np.einsum("bij,bj->bi", q_inv, u_albet)
        return u_dq

    def abc2dq(self, u_abc, eps):
        u_alpha_beta = u_abc @ self.t23.T 
        u_dq = self.albet2dq(u_alpha_beta, eps)
        return u_dq

    @staticmethod
    def apply_hex_constraint(u_albet):
        """Clip voltages in alpha/beta coordinates into the voltage hexagon"""
        u_albet_c = u_albet[:, 0] + 1j*u_albet[:, 1]
        idx = (np.sin(np.angle(u_albet_c)[..., np.newaxis] - 2/3*np.pi*np.arange(3)) >= 0).astype(int)
        rot_vecs = ROTATION_MAP[idx[:, 0], idx[:, 1], idx[:, 2]]
        np.multiply(u_albet_c, rot_vecs, out=u_albet_c)  # rotate sectors upwards
        np.clip(u_albet_c.real, - 2/3, 2/3, out=u_albet_c.real)
        np.clip(u_albet_c.imag, 0, 2/3 * np.sqrt(3), out=u_albet_c.imag)
        np.multiply(u_albet_c, np.conjugate(rot_vecs), out=u_albet_c)  # rotate back
        return np.column_stack([u_albet_c.real, u_albet_c.imag])

    def ode_step(self, u_dq, i_dq, l_dq, r_s, omega, psi_p):
        if self.saturated:
            # differential inductances
            # source: https://doi.org/10.1109/TII.2021.3060469
            q = self.t_dq_alpha_beta(self.tau * omega[..., np.newaxis])
            p_d = {q[:-4]: interp(i_dq)
                   for q, interp in self.interpolators.items()}
            L_diff = np.column_stack(
                [p_d[q] for q in ['L_dd', 'L_dq', 'L_qd', 'L_qq']]).reshape(-1, 2, 2)
            L_diff_inv = np.linalg.inv(L_diff)
            psi_dq = np.column_stack([p_d[psi] for psi in ['Psi_d', 'Psi_q']])
            di_dq_1 = np.einsum("bij,bj->bi",
                                (np.eye(2, 2)[np.newaxis, ...] - L_diff_inv*q*self.r_s_const*self.tau), i_dq)
            di_dq_2 = np.einsum("bij,bjk,bk->bi", L_diff_inv, q, u_dq*self.tau)
            di_dq_3 = np.einsum("bij,bjk,bk->bi",
                                L_diff_inv, (q - np.eye(2, 2)[np.newaxis, ...]), psi_dq)
            i_dq_k1 = di_dq_1 + di_dq_2 + di_dq_3
        else:
            # fundamental wave model
            # Source: https://doi.org/10.1109/IEMDC.2019.8785122
            q = np.array([[-r_s, omega*l_dq[:, 1]],
                         [-omega*(l_dq[:, 0]), -r_s]])
            di_dq = (u_dq + np.einsum("ijb,bj->bi", q, i_dq) - omega.reshape(-1, 1)
                     * np.column_stack([np.zeros_like(psi_p), psi_p])) / l_dq
            i_dq_k1 = i_dq + self.tau * di_dq  # explicit Euler
        return i_dq_k1

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        # If batchsize change, update the corresponding dimension
        self._batch_size = batch_size
        self.update_batch_dim()

    def generate_observation(self, u_dq):
        normalized_current_inc = self.i_dq / self.i_limit
        x_star = self.x_star[self.cntr, self.idx_sample, :]
        return np.hstack((
            normalized_current_inc,
            self.eps.reshape(-1, 1), x_star, u_dq/self.u_limit,
            #np.sin(self.eps).reshape(-1, 1), np.cos(self.eps).reshape(-1, 1),
            #u_dq/self.u_limit, x_star - normalized_current_inc,
            
        ))

    def step(self, u_dq_norm):
        #TODO Totzeit hinzufügen
        
        # clip in abc coordinates and unnormalize
        self.u_albet_norm = self.dq2albet(u_dq_norm, self.get_advanced_angle(
            self.eps, 0.5, self.tau, self.me_omega))
        self.u_albet_norm_clip = self.apply_hex_constraint(self.u_albet_norm)
        u_albet = self.u_albet_norm_clip * self.u_limit
        u_dq = self.albet2dq(u_albet, self.eps)

        # increment eps (rotor position)
        self.eps = self.step_eps(self.eps)

        # ode
        self.i_dq = self.ode_step(u_dq, self.i_dq, self.l_dq,
                                  self.r_s, self.el_omega, self.psi_p)

        # last x_star and state are irrelevant, they are not tracked nor processed later
        self.cntr = min(self.cntr + 1, len(self.x_star)-1)
        obs = self.generate_observation(u_dq)
        # bound check
        normalized_current_increment = self.i_dq / self.i_limit
        done = np.linalg.norm(normalized_current_increment, axis=1) > 1

        return obs, {}, done, {}

    def reset(self, shuffle_references=True, random_initial_values=False):
        """Reset the environment, return initial observation vector """
        self.eps = np.zeros(self.batch_size, dtype=np.float32)
        self.cntr = 0
        if random_initial_values:
            self.i_dq = np.random.rand(self.batch_size, 2).astype(np.float32)*2 - 1
        else:
            self.i_dq = np.zeros((self.batch_size, 2), dtype=np.float32)
        if shuffle_references:
            # Subtract 2x500 due to validation and evaluation set
            self.idx_sample = np.array(random.sample(
                range(500, self.x_star.shape[1]-500), self.batch_size))
        obs = self.generate_observation(np.zeros_like(self.i_dq))

        return obs, {}, False, {}
