"""
FrameTransformer Class for Multi-Frame 6D Pose/Vector Transformation

This class provides a generic and efficient solution for transforming pose, velocity, and wrench (force/torque)
data between arbitrary coordinate frames in a robot or vision system. The frame structure is user-definable as a tree,
supporting common robot frames (world, base, flange, tcp) as well as any number of eye-in-hand or eye-to-hand cameras.

Features:
---------
- Arbitrary 6D pose transformation between any two frames using a chain/tree-based architecture
- Compatible with hand-eye, eye-to-hand, and mixed vision/robot coordinate systems
- Supports velocity and wrench (force/torque) transformation with correct rotation and translation handling

Dependencies:
-------------
- numpy
- scipy.spatial.transform (SO(3) rotation/vector math)

Note:
- All poses must be [x, y, z, rx, ry, rz] where (rx, ry, rz) is a rotation vector (Rodrigues vector, not quaternion).
- For dynamic links (e.g. flange), always provide the current pose at runtime.
- Units and conventions must be consistent across your system.

Author: Yu-Peng, Yeh, 2025-07-08, nickyeh0415@gmail.com
"""

from scipy.spatial.transform import Rotation as R
import numpy as np

class FrameTransformer:
    def __init__(self, offset_base_world, offset_tcp_flange,
                 camera_eyeinhand_offsets=None, camera_eyetohand_offsets=None):
        """
        Initialize the FrameTransformer with a tree structure describing all frames and their static/dynamic offsets.
        All offsets are [x, y, z, rx, ry, rz], with (rx, ry, rz) in rotation vector (rotvec) format.

        Args:
            offset_base_world (list or np.ndarray): [6] pose of the base in the world frame (usually static, robot installation).
            offset_tcp_flange (list or np.ndarray): [6] pose of the TCP relative to the flange (usually static).
            camera_eyeinhand_offsets (dict): {id: [6]}, pose of each eye-in-hand camera relative to flange.
            camera_eyetohand_offsets (dict): {id: [6]}, pose of each eye-to-hand camera relative to world.
        """
        # Define the basic frame tree with parent relationships
        self.frames = {
            "world":  {"parent": None,        "offset": None},
            "base":   {"parent": "world",     "offset": offset_base_world},
            "flange": {"parent": "base",      "offset": None},  # This is dynamic, provided at runtime
            "tcp":    {"parent": "flange",    "offset": offset_tcp_flange},
        }

        # Add eye-in-hand cameras (relative to flange)
        if camera_eyeinhand_offsets:
            for cid, offset in camera_eyeinhand_offsets.items():
                self.frames[f"eyeinhand_{cid}"] = {"parent": "flange", "offset": offset}
        # Add eye-to-hand cameras (relative to world)
        if camera_eyetohand_offsets:
            for cid, offset in camera_eyetohand_offsets.items():
                self.frames[f"eyetohand_{cid}"] = {"parent": "world", "offset": offset}

    @staticmethod
    def get_pos_rotm(pose):
        """
        Convert a 6D pose [x, y, z, rx, ry, rz] into position and rotation matrix.

        Args:
            pose (array-like): 6D pose, where the last 3 elements are a rotation vector.

        Returns:
            tuple: (position (3,), rotation matrix (3,3))
        """
        pos = pose[:3]
        rotm = R.from_rotvec(pose[3:]).as_matrix()
        return pos, rotm

    @staticmethod
    def make_homogeneous_transform(pos, rotm):
        """
        Create a 4x4 homogeneous transformation matrix from position and rotation matrix.

        Args:
            pos (array-like): (3,) position vector
            rotm (array-like): (3,3) rotation matrix

        Returns:
            np.ndarray: (4,4) homogeneous transformation matrix
        """
        H = np.eye(4)
        H[:3, :3] = rotm
        H[:3, 3] = pos
        return H

    @staticmethod
    def combine_pos_rotm(position, rotation_matrix):
        """
        Combine a position and rotation matrix into a 6D pose [x, y, z, rx, ry, rz] (rotation vector).

        Args:
            position (array-like): (3,) position
            rotation_matrix (array-like): (3,3) rotation matrix

        Returns:
            np.ndarray: (6,) pose (with rotation vector)
        """
        rvec = R.from_matrix(rotation_matrix).as_rotvec()
        return np.concatenate((position, rvec))

    def _get_chain_to_root(self, frame, pose_flange_base=None):
        """
        Traverse from the specified frame up to the root ("world"),
        collecting each local-to-parent offset along the way.
        If the frame is "flange" and a dynamic pose_flange_base is provided,
        use it as the offset instead of the static value.

        Args:
            frame (str): The name of the starting frame.
            pose_flange_base (array-like): Current pose of flange in base frame (required if chain passes through flange).

        Returns:
            list of (frame, offset): Chain from the queried frame up to "world".
        """
        chain = []
        while frame is not None:
            parent = self.frames[frame]["parent"]
            offset = self.frames[frame]["offset"]
            if frame == "flange" and pose_flange_base is not None:
                offset = pose_flange_base
            chain.append((frame, offset))
            frame = parent
        return chain

    def _get_transform_matrix(self, from_frame, to_frame, pose_flange_base=None):
        """
        Compute the homogeneous transformation matrix from 'from_frame' to 'to_frame'.

        This works by:
        1. Finding the Lowest Common Ancestor (LCA) of the two frames.
        2. Chaining all local transforms from 'from_frame' up to the LCA (forward direction).
        3. Chaining all local transforms from 'to_frame' up to the LCA (inverted, backward direction).
        4. Composing these to obtain the overall transformation.

        Args:
            from_frame (str): Source frame.
            to_frame (str): Target frame.
            pose_flange_base (array-like): The dynamic pose of the flange in the base frame, if needed.

        Returns:
            np.ndarray: (4,4) homogeneous transformation matrix from 'from_frame' to 'to_frame'.
        """
        if from_frame == to_frame:
            return np.eye(4)
        chain_from = self._get_chain_to_root(from_frame, pose_flange_base)
        chain_to = self._get_chain_to_root(to_frame, pose_flange_base)
        frames_from = [f[0] for f in chain_from]
        frames_to = [f[0] for f in chain_to]
        # Find the lowest common ancestor (LCA)
        for f in frames_from:
            if f in frames_to:
                common = f
                break
        # Compose transforms from from_frame up to common
        T = np.eye(4)
        for frame, offset in chain_from:
            if frame == common: break
            if offset is None: continue
            pos, rot = self.get_pos_rotm(offset)
            T = self.make_homogeneous_transform(pos, rot) @ T
        # Compose transforms from to_frame up to common (inverse order)
        T_inv = np.eye(4)
        for frame, offset in chain_to:
            if frame == common: break
            if offset is None: continue
            pos, rot = self.get_pos_rotm(offset)
            T_inv = T_inv @ np.linalg.inv(self.make_homogeneous_transform(pos, rot))
        return T_inv @ T

    def transform_pose(self, pose, from_frame, to_frame, pose_flange_base=None):
        """
        Transform a 6D pose from one frame to another.

        Args:
            pose (array-like): [x, y, z, rx, ry, rz] in the 'from_frame'
            from_frame (str): Name of the input frame
            to_frame (str): Name of the target frame
            pose_flange_base (array-like): If the chain includes "flange", provide its current pose in base

        Returns:
            np.ndarray: (6,) pose in the 'to_frame'
        """
        T_total = self._get_transform_matrix(from_frame, to_frame, pose_flange_base)
        pos, rot = self.get_pos_rotm(pose)
        pos_new = T_total[:3, :3] @ pos + T_total[:3, 3]
        rot_new = T_total[:3, :3] @ rot
        return self.combine_pos_rotm(pos_new, rot_new)

    def _get_rotation_chain(self, from_frame, to_frame, tcp_pose_base=None, pose_flange_base=None):
        """
        Get the pure rotation matrix for transforming vectors from 'from_frame' to 'to_frame'.
        Useful for velocity, force, and other vector transformations.

        Args:
            from_frame (str): Source frame
            to_frame (str): Target frame
            tcp_pose_base, pose_flange_base: For compatibility with some APIs; only pose_flange_base is used here

        Returns:
            np.ndarray: (3,3) rotation matrix from 'from_frame' to 'to_frame'
        """
        T_total = self._get_transform_matrix(from_frame, to_frame, pose_flange_base)
        return T_total[:3, :3]

    def _get_translation_chain(self, from_frame, to_frame, tcp_pose_base=None, pose_flange_base=None):
        """
        Get the translation vector from 'from_frame' origin to 'to_frame', expressed in the 'to_frame'.

        Args:
            from_frame (str): Source frame
            to_frame (str): Target frame
            tcp_pose_base, pose_flange_base: For compatibility with some APIs; only pose_flange_base is used here

        Returns:
            np.ndarray: (3,) translation vector
        """
        T_total = self._get_transform_matrix(from_frame, to_frame, pose_flange_base)
        return T_total[:3, 3]

    def transform_velocity(self, velocity, tcp_pose_base, from_frame, to_frame, pose_flange_base=None):
        """
        Transform a spatial velocity vector from one frame to another (linear and angular, only rotation part).

        Args:
            velocity (array-like): [vx, vy, vz, wx, wy, wz] in 'from_frame'
            tcp_pose_base: (Unused in this function; included for API compatibility)
            from_frame (str): Source frame
            to_frame (str): Target frame
            pose_flange_base: (Optional) pose of flange in base if relevant

        Returns:
            np.ndarray: (6,) velocity in 'to_frame'
        """
        if from_frame == to_frame:
            return np.array(velocity)
        R_from_to = self._get_rotation_chain(from_frame, to_frame, tcp_pose_base, pose_flange_base)
        v = np.array(velocity[:3])
        w = np.array(velocity[3:])
        return np.concatenate((R_from_to @ v, R_from_to @ w))

    def transform_wrench(self, wrench, tcp_pose_base, from_frame, to_frame, pose_flange_base=None):
        """
        Transform a spatial wrench (force/torque) vector from one frame to another.
        The torque is adjusted for the new moment point if there is a translation between frames.

        Args:
            wrench (array-like): [Fx, Fy, Fz, Tx, Ty, Tz] in 'from_frame'
            tcp_pose_base: (Unused in this function; included for API compatibility)
            from_frame (str): Source frame
            to_frame (str): Target frame
            pose_flange_base: (Optional) pose of flange in base if relevant

        Returns:
            np.ndarray: (6,) wrench in 'to_frame'
        """
        if from_frame == to_frame:
            return np.array(wrench)
        R_from_to = self._get_rotation_chain(from_frame, to_frame, tcp_pose_base, pose_flange_base)
        p_from_to = self._get_translation_chain(from_frame, to_frame, tcp_pose_base, pose_flange_base)
        F = np.array(wrench[:3])
        T = np.array(wrench[3:])
        F_new = R_from_to @ F
        # Adjust torque for translation: T' = R*T + r x F'
        T_new = R_from_to @ T + np.cross(p_from_to, F_new)
        return np.concatenate((F_new, T_new))
