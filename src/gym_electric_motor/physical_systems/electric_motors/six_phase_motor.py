import math

import numpy as np

from .electric_motor import ElectricMotor
from scipy.linalg import block_diag

class SixPhaseMotor(ElectricMotor):
    """
    The SixPhaseMotor and its subclasses implement the technical system of Six Phase Motors.

    """

    # transformation matrix from abc to alpha-beta representation
    # VSD is the modeling approach used for multi-phase machines, which represents a generalized form of the clarke transformation
    # for the six phase machine with Winding configuration in the stator- and rotor-fixed coordinate systems with δ =(2/3)*π and γ =π/6
    _t46= 1/ 3 * np.array([
        [1, -0.5, -0.5, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0],
        [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0.5, 0.5, -1],
        [1, -0.5, -0.5, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0],
        [0, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0.5, 0.5, -1]
    ])

    # i_s - stator current
    @staticmethod
    def t_46(quantities):
        """
        Transformation from abc representation to alpha-beta representation

        Args:
            quantities: The properties in the abc representation like ''[i_sa1, i_sb1, i_sc1, i_sa2, i_sb2, i_sc2]''

        Returns:
            The converted quantities in the alpha-beta representation like ''[i_salpha, i_sbeta, i_sX, i_sY]''
        """
        return np.matmul(SixPhaseMotor._t46, quantities)

    @staticmethod
    def q(quantities, epsilon):
        """
        Transformation of the abc representation into dq using the electrical angle

        Args:
            quantities: the properties in the abc representation like ''[i_sa1, i_sb1, i_sc1, i_sa2, i_sb2, i_sc2]''
            epsilon: electrical rotor position

        Returns:
            The converted quantities in the dq representation like ''[i_sd, i_sq, i_sx, i_sy, i_s0+, i_s0-]''.
            since 2N topology is considered (case where the neutral points are not connected) i_s0+, i_s0- will not be taken into account 
        """
        """
        t_vsd = 1/ 3 * np.array([
            [1, -0.5, -0.5, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0],
            [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0.5, 0.5, -1],
            [1, -0.5, -0.5, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0],
            [0, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0.5, 0.5, -1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1]
        ])
        tp_alphaBetaXY = np.array([
                 [cos, sin, 0, 0, 0, 0],
                 [-sin, cos, 0, 0, 0, 0],
                 [0, 0, cos, -sin, 0, 0],
                 [0, 0, sin, cos, 0, 0],
                 [0, 0, 0, 0, 1, 0]
                 [0, 0, 0, 0, 0, 1]
        ])
        """
        t_vsd = 1/ 3 * np.array([
            [1, -0.5, -0.5, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0],
            [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0.5, 0.5, -1],
            [1, -0.5, -0.5, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0],
            [0, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0.5, 0.5, -1],
        ])

        def rotation_matrix(theta):
           return np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
        ])
        
        t1 = rotation_matrix(epsilon)
        t2 = rotation_matrix(-epsilon)
        tp_alphaBetaXY = block_diag(t1,t2)
        tp_vsd = np.matmul(tp_alphaBetaXY, t_vsd)
        return np.matmul(tp_vsd, quantities)
       
    
    @staticmethod
    def q_inv(quantities, epsilon):
        """
        Transformation of the dq representation into abc

        Args:
            quantities: the properties in the dq representation like ''[i_sd, i_sq, i_sx, i_sy, i_s0+, i_s0-]''.
            epsilon: electrical rotor position

        Returns:
            The converted quantities in the abc representation like ''[i_sa1, i_sb1, i_sc1, i_sa2, i_sb2, i_sc2]''.

        """
        t_vsd = 1/ 3 * np.array([
            [1, -0.5, -0.5, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0],
            [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3), 0.5, 0.5, -1],
            [1, -0.5, -0.5, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0],
            [0, -0.5 * np.sqrt(3), 0.5 * np.sqrt(3), 0.5, 0.5, -1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1]
        ])
        cos = math.cos(epsilon)
        sin = math.sin(epsilon)
        tp_alphaBetaXY = np.array([
                 [cos, sin, 0, 0, 0, 0],
                 [-sin, cos, 0, 0, 0, 0],
                 [0, 0, cos, -sin, 0, 0],
                 [0, 0, sin, cos, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1],
        ])
        tp_vsd = np.matmul(tp_alphaBetaXY, t_vsd)
        inv_tpVsd = np.linalg.inv(tp_vsd)
        modified_inv_tpVsd = np.delete(inv_tpVsd, [4, 5], axis=1)
        return np.matmul(modified_inv_tpVsd, quantities)

  