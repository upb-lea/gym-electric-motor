default_motor_parameter = {
    'DcSeries': [
        # Series Set 0 (default)
        {
            'r_a': 2.78, 'r_e': 1.0, 'l_a': 6.3e-3, 'l_e': 1.6e-3, 'l_e_prime': 0.05, 'j': 0.017, 'u_sup': 420,
            'i_N': 50, 'u_N': 420, 'omega_N': 368, 'torque_N': 250
        },
        # Series Set 1
        {
            'r_a': 0.9, 'r_e': 20, 'l_a': 1e-1, 'l_e': 1e-1, 'l_e_prime': 3.5, 'j': 0.007, 'u_sup': 42, 'u_N': 42,
            'i_N': 1.6, 'omega_N': 60, 'torque_N': 92
        },
    ],
    'DcShunt': [
        # Shunt Set 0 (default)
        {
            'r_a': 2.78, 'r_e': 350, 'l_a': 6.3e-3, 'l_e': 160, 'l_e_prime': 0.94, 'j': 0.017, 'u_sup': 420,
            'i_a_N': 50, 'i_e_N': 1.2, 'u_N': 420, 'omega_N': 368, 'torque_N': 40
        },
    ],
    'DcPermEx': [
        # PermEx Set 0 (default)
        {
            'r_a': 25.0, 'r_e': 258.0, 'l_a': 3.438e-2, 'l_e': 0.5, 'l_e_prime': 0.78, 'j': 1.0, 'u_sup': 400,
            'u_N': 420, 'i_N': 16, 'omega_N': 22, 'torque_N': 280, 'psi_e': 18
        }
    ],
    'DcExtEx': [
        # ExtEx Set 0 (default)
        {
            'r_a': 0.78, 'r_e': 350, 'l_a': 6.3e-3, 'l_e': 60, 'l_e_prime': 0.94, 'j': 0.017, 'u_sup': 420,
            'i_a_N': 50, 'i_e_N': 1.2, 'u_N': 420, 'u_a_N': 420, 'u_e_N': 420, 'omega_N': 368, 'torque_N': 40
        },
    ]
}

default_load_parameter = [
    # Load Set 0
    {
     'a': 0.01, 'b': 0.12, 'c': 0.1, 'J_load': 1.0
    },
    # Load Set 1 (IDLE Load)
    {
     'a': 0, 'b': 0, 'c': 0, 'J_load': 0
    },
]
