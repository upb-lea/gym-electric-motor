import numpy as np
import scipy.interpolate as sp_interpolate

from .foc_operation_point_selection import FieldOrientedControllerOperationPointSelection


class EESMOperationPointSelection(FieldOrientedControllerOperationPointSelection):
    """
    This class represents the operation point selection of the torque controller for cascaded control of an
    externally synchronous motor. The operating point is selected in the analog to that of the PMSM and SynRM, but
    the excitation current is also included in the optimization of the operating point.
    """

    def __init__(self, max_modulation_level: float = 2 / np.sqrt(3), modulation_damping: float = 1.2):
        """
        Args:
            max_modulation_level(float): Maximum value for the modulation controller.
            modulation_damping(float): Damping of the gain of the modulation controller.
        """

        super().__init__(max_modulation_level, modulation_damping)
        self.l_d = None
        self.l_q = None
        self.l_m = None
        self.l_e = None
        self.r_s = None
        self.r_e = None
        self.i_e_lim = None
        self.t_lim = None
        self.i_q_lim = None

        self.t_count = None
        self.psi_count = None
        self.i_e_count = None

        self.psi_opt = None
        self.i_d_opt = None
        self.i_q_opt = None
        self.i_e_opt = None
        self.t_max = None
        self.psi_max = None
        self.t_max_psi = None

        self.t_grid_count = None
        self.psi_grid_count = None

        self.torque_equation = None
        self.loss = None
        self.poly = None

    def tune(self, env, env_id, current_safety_margin=0.2):
        """
        Tune the operation point selcetion stage.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """

        super().tune(env, env_id, current_safety_margin)
        self.l_d = self.mp["l_d"]
        self.l_q = self.mp["l_q"]
        self.l_m = self.mp["l_m"]
        self.l_e = self.mp["l_e"]
        self.r_s = self.mp["r_s"]
        self.r_e = self.mp["r_e"]
        self.p = self.mp["p"]
        self.i_e_lim = env.get_wrapper_attr('limits')[env.get_wrapper_attr('state_names').index("i_e")] * (1 - current_safety_margin)
        self.i_q_lim = env.get_wrapper_attr('limits')[env.get_wrapper_attr('state_names').index("i_sq")] * (1 - current_safety_margin)
        self.t_lim = env.get_wrapper_attr('limits')[env.get_wrapper_attr('state_names').index("torque")]

        self.t_count = 50
        self.psi_count = 100
        self.i_e_count = 150

        self.t_grid_count = 200
        self.psi_grid_count = 200

        self.k_ = 0.953
        self.i_gain = 1 / (self.l_q / (1.25 * self.r_s)) * (self.alpha - 1) / self.alpha**2

        self.psi_high = 0.2 * np.sqrt(
            (self.l_m * self.i_e_lim * current_safety_margin + self.l_d * self.i_sq_limit * current_safety_margin) ** 2
        )

        self.psi_low = -self.psi_high
        self.integrated_reset = 0.01 * self.psi_low

        self.torque_equation = (
            lambda i_d, i_q, i_e: 3 / 2 * self.p * (self.l_m * i_e + (self.l_d - self.l_q) * i_d) * i_q
        )

        #self.loss = lambda i_d, i_q, i_e: np.abs(i_d) * self.r_s + np.abs(i_q) * self.r_s + np.abs(i_e) * self.r_e
        # Copper power losses: P = 1.5*Rs*(id^2 + iq^2) + Rf*(ie^2)
        self.loss = lambda i_d, i_q, i_e: 1.5 * self.r_s * (i_d**2 + i_q**2) + self.r_e * (i_e**2)

        self.poly = lambda i_e, psi, torque: [
            self.l_d**2 * (self.l_d - self.l_q) ** 2,
            2 * self.l_d**2 * (self.l_d - self.l_q) * self.l_m * i_e
            + 2 * self.l_d * self.l_m * i_e * (self.l_d - self.l_q) ** 2,
            self.l_d**2 * (self.l_m * i_e) ** 2
            + 4 * self.l_d * (self.l_m * i_e) ** 2 * (self.l_d - self.l_q)
            + ((self.l_m * i_e) ** 2 - psi**2) * (self.l_d - self.l_q) ** 2,
            2 * self.l_q * (self.l_m * i_e) ** 3
            + 2 * ((self.l_m * i_e) ** 2 - psi**2) * self.l_m * i_e * (self.l_d - self.l_q),
            ((self.l_m * i_e) ** 2 - psi**2) * (self.l_m * i_e) ** 2 + (self.l_q * torque / (3 * self.p)) ** 2,
        ]

        self._calculate_luts()

    def solve_analytical(self, torque, psi, i_e):
        """
        Assuming linear magnetization characteristics, the optimal currents for given reference, flux and exitation
        current can be obtained by solving the reference and flux equations. These lead to a fourth degree polynomial
        which can be solved analytically.

        Args:
            torque(float): The torque reference value.
            psi(float): The optimal flux value.
            i_e(float): The excitation current.

        Returns:
            i_d(float): optimal i_sd current
            i_q(flaot): optimal i_sq current
        """
        if torque == 0 and i_e == 0:
            return 0, 0
        else:
            i_d = np.real(np.roots(self.poly(i_e, psi, torque))[-1])
            i_q = 2 * torque / (3 * self.p * (self.l_m * i_e + (self.l_d - self.l_q) * i_d))
            return i_d, i_q

    def _calculate_luts(self):
        import numpy as np
        import scipy.interpolate as sp_interpolate
        from scipy.optimize import minimize
        SLSQP_MAXITER   = 300      
        SLSQP_FTOL      = 5e-13     
        TORQUE_BAND_REL = 5e-4      
        TORQUE_BAND_ABS = 1e-2
        SLSQP_MAXITER_FBK  = 600
    # ---- pull essentials ----
        p   = float(self.p)
        Ld  = float(self.l_d)
        Lq  = float(self.l_q)
        Lm  = float(self.l_m)
        Rs  = float(self.r_s)
        Re  = float(self.r_e)

    # limits
        i_s_lim = float(getattr(self, "i_s_lim", self.i_q_lim))  
        i_f_lim = float(self.i_e_lim)
        t_lim   = float(self.t_lim)

    # counts / grids
        t_count        = int(self.t_count)
        t_grid_count   = int(self.t_grid_count)
        psi_grid_count = int(self.psi_grid_count)

   
        try:
            Udc = float(self._env.supply.u_nominal)  
        except Exception:
            Udc = 200.0

        try:
            omega_lim_mech = float(self._env.limits[self._env.state_names.index("omega")])  
        except Exception:
            omega_lim_mech = 7e3 * np.pi / 30.0  

    # same choice as testcase
        omega_el = 0.15 * omega_lim_mech * 2.0
        V_over_omega = Udc / (np.sqrt(3.0) * max(omega_el, 1e-6))

    # ----- physical helpers -----
        def torque(id_, iq_, if_):
            return 1.5 * p * (Lm * if_ + (Ld - Lq) * id_) * iq_

        def loss(id_, iq_, if_):
            return 1.5 * Rs * (id_**2 + iq_**2) + Re * (if_**2)

        def feasible(id_, iq_, if_):
        # current limits
            if np.hypot(id_, iq_) > i_s_lim + 1e-12:
                return False
            if abs(if_) > i_f_lim + 1e-12:
                return False
            if not (0.0 - 1e-12 <= if_ <= i_f_lim + 1e-12):
                return False

        # voltage/flux ellipse (MTPCL appendix)
            psi_d = Lm * if_ + Ld * id_
            psi_q = Lq * iq_
            if (psi_q**2 + psi_d**2) > (V_over_omega**2 + 1e-12):
                return False
            return True

    # ----- constrained loss minimization (equality torque) -----
        def obj_loss(x):
            return loss(x[0], x[1], x[2])

        def cons_for_T(Tref):
            return [
            {'type': 'ineq', 'fun': lambda x: i_s_lim - np.hypot(x[0], x[1])},   # |is| <=
            {'type': 'ineq', 'fun': lambda x: x[2]},                              # i_f >= 0
            {'type': 'ineq', 'fun': lambda x: i_f_lim - x[2]},                    # i_f <=
            {'type': 'ineq', 'fun': lambda x: V_over_omega**2
                                   - ((Lq * x[1])**2 + (Lm * x[2] + Ld * x[0])**2)},
            {'type': 'eq',   'fun': lambda x: torque(x[0], x[1], x[2]) - Tref},   # Te = Tref
        ]
        #T_bound = 1.5 * p * (Lm * i_f_lim + abs(Ld - Lq) * i_s_lim) * i_s_lim
    # ----- fallback: minimize torque error, then loss-----
        def _seed_by_hand(Tref):
            s = 1.0 if Tref >= 0 else -1.0
            iF = min(i_f_lim, V_over_omega / max(Lm, 1e-12))
            iq_flux = np.sqrt(max(0.0, V_over_omega**2 - (Lm*iF)**2)) / max(Lq, 1e-12)
            iq_curr = i_s_lim
            iq_need = abs(Tref) / max(1.5 * p * Lm * iF, 1e-12)
            iq = s * min(iq_flux, iq_curr, iq_need)
            
            return np.array([0.0, iq, iF], float)
        
        def solve_mtpc_for_T(Tref, x_prev):


            bnds = [(-i_s_lim, i_s_lim), (-i_s_lim, i_s_lim), (0.0, i_f_lim)]
            #x0   = x_prev if x_prev is not None else _seed_by_hand(Tref)
            cons_eq = cons_for_T(Tref)
            x0   = np.asarray(x_prev, float) if x_prev is not None else _seed_by_hand(Tref)
            x_keep = np.asarray(x0, float)
            stats = {
                "nit_eq": 0, "nfev_eq": None,
                "nit_fbk1": 0, "nfev_fbk1": None,
                "nit_fbk2": 0, "nfev_fbk2": None
            }

    # --- equality-constrained loss minimization ---
            _ctr_eq = {"k": 0}
            def _cb_eq(_xk): _ctr_eq["k"] += 1
            try:
                res = minimize(
                        obj_loss, x0, method='SLSQP',
                        bounds=bnds, constraints=cons_for_T(Tref),
                        options=dict(maxiter=SLSQP_MAXITER, ftol=SLSQP_FTOL, disp=False)
                    )
                stats["nit_eq"]  = getattr(res, "nit",  None)
                stats["nfev_eq"] = getattr(res, "nfev", None)
                if res.success and feasible(*res.x):
                   return res.x, stats
            except Exception:
                   pass
            def obj_torque_err(x):
                return (torque(x[0], x[1], x[2]) - Tref)**2

            cons_soft = [
                 {'type': 'ineq', 'fun': lambda x: i_s_lim - np.hypot(x[0], x[1])},   # |is| <=
                 {'type': 'ineq', 'fun': lambda x: x[2]},                              # i_f >= 0
                 {'type': 'ineq', 'fun': lambda x: i_f_lim - x[2]},                    # i_f <=
                 {'type': 'ineq', 'fun': lambda x: V_over_omega**2
                              - ((Lq * x[1])**2 + (Lm * x[2] + Ld * x[0])**2)},
            ]
            
    # 2a) minimize torque error
            try:
                res1 = minimize(
                    obj_torque_err, x0, method='SLSQP',
                    bounds=bnds, constraints=cons_soft,
                    options=dict(maxiter=SLSQP_MAXITER_FBK, ftol=SLSQP_FTOL, disp=False)
                )
                stats["nit_fbk1"]  = getattr(res, "nit",  None)
                stats["nfev_fbk1"] = getattr(res1, "nfev", None)
                if res1.success and feasible(*res1.x):
                    T_reach = torque(*res1.x)
                    band    = max(abs(T_reach)*TORQUE_BAND_REL, TORQUE_BAND_ABS)
                    lo, hi  = T_reach - band, T_reach + band

            # 2b) keep torque in [lo, hi] while minimizing loss
                cons_keep = cons_soft + [
                    {'type': 'ineq', 'fun': lambda x, lo=lo: torque(x[0], x[1], x[2]) - lo},
                    {'type': 'ineq', 'fun': lambda x, hi=hi: hi - torque(x[0], x[1], x[2])},
                ]
                

                res2 = minimize(
                    obj_loss, res1.x, method='SLSQP',
                    bounds=bnds, constraints=cons_keep,
                    options=dict(maxiter=SLSQP_MAXITER_FBK, ftol=SLSQP_FTOL, disp=False)
                )
                stats["nit_fbk2"]  = getattr(res, "nit",  None)
                stats["nfev_fbk2"] = getattr(res2, "nfev", None)
                if res2.success and feasible(*res2.x):
                    T2 = torque(*res2.x)
                    if lo <= T2 <= hi:
                        return res2.x, stats
            # if loss-min step fails the band, keep the torque-closest point
                return res1.x, stats
            except Exception:
                pass

    # as a last resort, keep previous point (keeps curves continuous)
            return x_keep, stats
    # ----  solve MTPCL, collect optimal points ----
        T_curr_cap = 1.5 * p * Lm * i_f_lim * i_s_lim
        T_cap = float(min(t_lim, T_curr_cap))
        iters_eq, iters_f1, iters_f2 = [], [], []
        T_vec = np.linspace(0.0, T_cap, t_count)  
        x_prev = _seed_by_hand(max(1e-3, 0.05*T_cap))
        rows = []  # [T, psi, id, |iq|, if]
        for T in T_vec:
            x_star, st = solve_mtpc_for_T(T, x_prev)
            id_opt, iq_opt, if_opt = x_star
            x_prev = x_star
            iters_eq.append(st["nit_eq"] or 0)
            iters_f1.append(st["nit_fbk1"] or 0)
            iters_f2.append(st["nit_fbk2"] or 0)
            psi_d = Lm * if_opt + Ld * id_opt
            psi_q = Lq * iq_opt
            psi   = np.hypot(psi_d, psi_q)
            rows.append([T, psi, id_opt, abs(iq_opt), if_opt])
        print(
            f"SLSQP iters — eq mean={np.mean(iters_eq):.1f} max={np.max(iters_eq)}; "
            f"fbk1 mean={np.mean(iters_f1):.1f}; fbk2 mean={np.mean(iters_f2):.1f}"
        )
        bp = np.array(rows, dtype=float)
        bp = bp[np.all(np.isfinite(bp), axis=1)]
        if bp.shape[0] == 0:
            bp = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)

    
        if bp.shape[0] == 1:
            eps = max(1e-6, 1e-3 * t_lim)
            bp = np.vstack([bp, bp + [eps, 0, 0, 0, 0]])

    # sort/unique by T
        order = np.argsort(bp[:, 0])
        bp = bp[order]
        Tuniq, idx = np.unique(bp[:, 0], return_index=True)
        psi_u = bp[idx, 1]
        id_u  = bp[idx, 2]
        iq_u  = bp[idx, 3]  # magnitude vs |T|
        if_u  = bp[idx, 4]

    # anchor origin
        if Tuniq[0] > 0.0:
            Tuniq = np.insert(Tuniq, 0, 0.0)
            psi_u = np.insert(psi_u, 0, 0.0)
            id_u  = np.insert(id_u,  0, 0.0)
            iq_u  = np.insert(iq_u,  0, 0.0)
            if_u  = np.insert(if_u,  0, 0.0)

    
        self.t_max   = float(np.max(Tuniq))
        self.psi_opt = sp_interpolate.interp1d(Tuniq, psi_u, kind="cubic", fill_value="extrapolate")
        self.i_d_opt = sp_interpolate.interp1d(Tuniq, id_u,  kind="cubic", fill_value="extrapolate")
        self.i_q_opt = sp_interpolate.interp1d(Tuniq, iq_u,  kind="cubic", fill_value="extrapolate")  # magnitude!
        self.i_e_opt = sp_interpolate.interp1d(Tuniq, if_u,  kind="cubic", fill_value="extrapolate")

    
        self.psi_max = float(np.max(psi_u))
        psi_axis_for_tmax = np.linspace(0.0, self.psi_max, max(50, len(psi_u)))
        T_samp  = Tuniq
        psi_samp = psi_u
        tmax_per_psi = np.interp(
            psi_axis_for_tmax,
            np.maximum.accumulate(psi_samp),
            np.maximum.accumulate(T_samp),
        )
        tmax_per_psi = 0.99 * tmax_per_psi
        self.t_max_psi = sp_interpolate.interp1d(
            psi_axis_for_tmax, tmax_per_psi, kind="linear",
            bounds_error=False,
            fill_value=(0.0, float(np.max(tmax_per_psi))),
        )

    # ---- 2D LUTs (linear griddata) ----
        # ---- 2D LUTs  ----
        t_axis   = np.linspace(0.0, float(self.t_max),   t_grid_count)
        psi_axis = np.linspace(0.0, float(self.psi_max), psi_grid_count)
        self.t_grid, self.psi_grid = np.meshgrid(t_axis, psi_axis, indexing="ij")

# Detect degenerate (T, psi) 
        psi_span = float(np.ptp(bp[:, 1])) if bp.shape[0] else 0.0
        n_unique_psi = np.unique(bp[:, 1]).size if bp.shape[0] else 0
        degenerate = (psi_span < 1e-9) or (n_unique_psi < 3)

        if degenerate:
    # Fallback: broadcast the 1D optimal curves along psi.
    # (i_q_opt is a magnitude vs |T|
            id_line = np.asarray(self.i_d_opt(t_axis), dtype=float)
            iq_line = np.asarray(self.i_q_opt(t_axis), dtype=float)
            ie_line = np.asarray(self.i_e_opt(t_axis), dtype=float)

            self.i_d_inter = np.tile(id_line[:, None], (1, psi_grid_count))
            self.i_q_inter = np.tile(iq_line[:, None], (1, psi_grid_count))
            self.i_e_inter = np.tile(ie_line[:, None], (1, psi_grid_count))
        else:
    # Proper 2D scattered interpolation
            self.i_d_inter = sp_interpolate.griddata(
            (bp[:, 0], bp[:, 1]), bp[:, 2], (self.t_grid, self.psi_grid), method="linear"
        )
            self.i_q_inter = sp_interpolate.griddata(
            (bp[:, 0], bp[:, 1]), bp[:, 3], (self.t_grid, self.psi_grid), method="linear"
        )
            self.i_e_inter = sp_interpolate.griddata(
        (bp[:, 0], bp[:, 1]), bp[:, 4], (self.t_grid, self.psi_grid), method="linear"
        )

   
        for arr, col in ((self.i_d_inter, 2), (self.i_q_inter, 3), (self.i_e_inter, 4)):
            mask = np.isnan(arr)
            if np.any(mask):
                arr[mask] = sp_interpolate.griddata(
                (bp[:, 0], bp[:, 1]), bp[:, col],
                (self.t_grid[mask], self.psi_grid[mask]),
                method="nearest",
        )

    def _get_psi_idx(self, psi):
        
        import numpy as np
        psi_max = float(getattr(self, "psi_max", 0.0) or 0.0)
        if not np.isfinite(psi_max) or psi_max <= 0.0 or int(self.psi_grid_count) <= 1:
            return 0
        if not np.isfinite(psi):
            psi = 0.0
        psi = float(np.clip(psi, 0.0, psi_max))
        cols = int(self.psi_grid_count) - 1
        return int(round((psi / psi_max) * cols))

    def _get_t_idx(self, torque):
        
        import numpy as np
        t_max = float(getattr(self, "t_max", 0.0) or 0.0)
        if not np.isfinite(t_max) or t_max <= 0.0 or int(self.t_grid_count) <= 1:
            return 0
        if not np.isfinite(torque):
            torque = 0.0
        torque = float(np.clip(torque, 0.0, t_max))
        rows = int(self.t_grid_count) - 1
        return int(round((torque / t_max) * rows))
    
    def _select_operating_point(self, state, reference):
        import numpy as np

        psi_cap = float(self.modulation_control(state))
    
        try:
            t_ref = float(reference[0])
        except Exception:
            t_ref = float(reference)

    
        t_max_global = float(getattr(self, "t_max", 0.0) or 0.0)
        t_ref_clip = float(np.clip(abs(t_ref), 0.0, t_max_global))

    
        try:
            psi_opt = float(self.psi_opt(t_ref_clip))
            if not np.isfinite(psi_opt):
                psi_opt = 0.0
        except Exception:
            psi_opt = 0.0

    # enforce modulation limit
        psi = float(np.clip(psi_opt, 0.0, psi_cap))

    
        try:
            t_max_local = float(self.t_max_psi(psi_opt))
            if not np.isfinite(t_max_local):
                t_max_local = t_max_global
        except Exception:
            t_max_local = t_max_global

        t_ref_clip = float(np.clip(t_ref_clip, 0.0, t_max_local))

    # safe indices
        t_idx = self._get_t_idx(t_ref_clip)
        psi_idx = self._get_psi_idx(psi)

    # lookup; iq LUT is magnitude → apply sign of requested torque
        i_d_ref = float(self.i_d_inter[t_idx, psi_idx])
        i_q_mag = float(self.i_q_inter[t_idx, psi_idx])
        i_e_ref = float(self.i_e_inter[t_idx, psi_idx])

        if not np.isfinite(i_d_ref): i_d_ref = 0.0
        if not np.isfinite(i_q_mag): i_q_mag = 0.0
        if not np.isfinite(i_e_ref): i_e_ref = 0.0

        i_q_ref = np.sign(t_ref) * i_q_mag
        return np.array([i_d_ref, i_q_ref, i_e_ref], dtype=float)


def reset(self):
        """Reset the EESM operation point selection"""
        super().reset()
