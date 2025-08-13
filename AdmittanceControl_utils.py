"""
AdmittanceControl Class for 6-DOF Translational/Rotational Admittance

This class implements a discrete-time admittance controller that maps external
forces/torques to virtual accelerations (or angular accelerations), and provides
Euler/Semi-implicit integration utilities for velocity/pose updates.

Features
--------
- Decoupled 3×3 diagonal virtual mass–damping–stiffness for translation & rotation

Dependencies
------------
- numpy
- scipy.spatial.transform.Rotation (SO(3) log/exp helpers)
- scipy.linalg.expm (matrix exponential)

Note
----
- x, v, a are 3×1 vectors in the same (base/world) frame.
- R_* are 3×3 rotation matrices mapping from body to base/world.
- Force/torque inputs F_ext/Tau_ext are in base/world frame.
- Damping choice:
    If K > 0, a common setting is D = 2 ζ ∘ sqrt(M ∘ K) (element-wise on diagonals).
    If K = 0, set D directly (do NOT derive from ζ and K).
- All matrices here are assumed diagonal for simplicity; extend to full 3×3 if needed.

Author: Yu-Peng, Yeh, 2025-07-08, nickyeh0415@gmail.com
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

class AdmittanceControl:
    def __init__(self, M_trans = None, K_trans = None, Zeta_trans = None,
                 M_rot = None, K_rot = None, Zeta_rot = None):
        
        """
        Initialize admittance control parameters for translation and rotation.

        Parameters
        ----------
        M_trans, K_trans, Zeta_trans : list or None
            Mass, stiffness, and damping ratio for translational DOFs
        M_rot, K_rot, Zeta_rot : list or None
            Mass, stiffness, and damping ratio for rotational DOFs
        """

        self.M_trans = np.diag(M_trans if M_trans is not None else [50.0, 50.0, 50.0])
        self.K_trans = np.diag(K_trans if K_trans is not None else [0.0, 0.0, 0.0])
        self.Zeta_trans = np.diag(Zeta_trans if Zeta_trans is not None else [1.0, 1.0, 1.0])

        self.M_rot = np.diag(M_rot if M_rot is not None else [50.0, 50.0, 50.0])
        self.K_rot = np.diag(K_rot if K_rot is not None else [0.0, 0.0, 0.0])
        self.Zeta_rot = np.diag(Zeta_rot if Zeta_rot is not None else [1.0, 1.0, 1.0])

    @staticmethod
    def skew(v):
        """Returns the skew-symmetric matrix of a 3D vector."""
        v = np.asarray(v).flatten()  # Ensure v is a 1D NumPy array
        return np.array([
            [  0,   -v[2],  v[1]],
            [ v[2],   0,   -v[0]],
            [-v[1],  v[0],   0]
        ])
    
    def translational(self, F_ext, x_des, x_curr, v_des, v_curr):
        """
        Compute translational acceleration based on admittance control law.

        Parameters
        ----------
        F_ext : np.ndarray
            External force vector (3x1) in base/world frame
        x_des, x_curr : np.ndarray
            Desired and current position vector (3x1) in base/world frame
        v_des, v_curr : np.ndarray
            Desired and current velocity vector (3x1) in base/world frame

        Returns
        -------
        a_current : np.ndarray
            Computed translational acceleration vector (3x1)
        """
        if np.allclose(self.K_trans, 0):
            D_trans = 2 * self.Zeta_trans * np.sqrt(self.M_trans)
        else:
            D_trans = 2 * self.Zeta_trans * np.sqrt(self.K_trans * self.M_trans)

        f_error = F_ext - D_trans @ (v_curr - v_des) - self.K_trans @ (x_curr - x_des)
        a_current = np.linalg.solve(self.M_trans, f_error)

        return a_current
    
    def rotational(self, Tau_ext, r_des, r_mes, w_des, w_curr):
        """
        Compute rotational acceleration using admittance law.

        Parameters
        ----------
        Tau_ext : np.ndarray
            External torque vector (3x1) in base/world frame    
        r_des, r_mes : np.ndarray
            Desired and measured rotation matrices (3x3) mapping from body to base/world frame
        w_des, w_curr : np.ndarray
            Desired and current angular velocity vectors (3x1) in base/world frame

        Returns
        -------
        alpha_current : np.ndarray
            Computed rotational acceleration vector (3x1)
        """
        if np.allclose(self.K_rot, 0):
            D_rot = 2 * self.Zeta_rot * np.sqrt(self.M_rot)
        else:
            D_rot = 2 * self.Zeta_rot * np.sqrt(self.K_rot * self.M_rot)

        r_error_matrix = r_des.T @ r_mes
        r_error_vec = R.from_matrix(r_error_matrix).as_rotvec()
        r_error_vec_base = (r_des @ r_error_vec).reshape(3, 1)

        tau_error = Tau_ext - D_rot @ (w_curr - w_des) - self.K_rot @ (r_error_vec_base)
        # print(tau_error)
        alpha_current = np.linalg.solve(self.M_rot, tau_error)

        return alpha_current
    
    def trans_integrate(self, x, v, a, dt, mode="velocity"):
        """
        Integrate translational velocity and position using semi-implicit Euler method.
        
        Parameters
        ----------
        x, V : np.ndarray
            Current position and velocity vectors (3x1) in base/world frame
        a : np.ndarray
            Current translational acceleration vector (3x1)
        dt : float
            Time step for integration
        mode : str
            Integration mode, either "velocity" or "position"

        Returns
        -------
        v_new : np.ndarray
            New velocity vector (3x1) in base/world frame
        x_new : np.ndarray, optional
            New position vector (3x1) in base/world frame, only if mode is "position"
        """
        v_new = v + a * dt

        if mode == "velocity":
            return v_new
        elif mode == "position":
            x_new = x + v * dt
            return v_new, x_new
        else:
            raise ValueError("mode must be 'velocity' or 'position'")
        
    def rot_integrate(self, r_d, r, w, alpha, dt, mode="velocity"):
        """
        Integrate rotational velocity and pose using semi-implicit Euler method.

        Parameters
        ----------
        r_d : np.ndarray
            Desired rotation matrix (3x3) mapping from body to base/world frame
        r : np.ndarray
            Current rotation matrix (3x3) mapping from body to base/world frame
        w : np.ndarray
            Current angular velocity vector (3x1) in base/world frame
        alpha : np.ndarray
            Current rotational acceleration vector (3x1)
        dt : float
            Time step for integration
        mode : str
            Integration mode, either "velocity" or "position"

        Returns
        -------
        w_tcp : np.ndarray
            New angular velocity vector (3x1) in TCP frame
        r_current : np.ndarray, optional
            New rotation matrix (3x3) mapping from body to base/world frame, only if mode is "position"
        rotation_vector : np.ndarray, optional
            New rotation vector (3x1) in base/world frame, only if mode is "position"
        """
        
        w_new = w + alpha * dt
        w_tcp = r_d.T @ w_new

        if mode == "velocity":
            return w_tcp
        elif mode == "position":
            r_current = r @ expm(self.skew(w_tcp * dt))
            rotation_vector = R.from_matrix(r_current).as_rotvec()
            return w_new, r_current, rotation_vector
        else:
            raise ValueError("mode must be 'velocity' or 'position'")        




