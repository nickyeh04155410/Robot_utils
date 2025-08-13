from frame_transformer import FrameTransformer
from RTDE_utils import connect_rtde, check_and_reconnect

import numpy as np
import multiprocessing

offset_base_world = [0, 0, 0, 0, 0, 0]   # base in world
offset_tcp_flange  = [0.0, -0.09417, 0.0825, 0.0, 0.0, 0.0]
pose_in_tcp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ft = FrameTransformer(offset_base_world, offset_tcp_flange)

# print("== tcp → base ==")
# pose_base = ft.transform_pose(pose_in_tcp, from_frame="tcp", to_frame="base", pose_flange_base=pose_flange_base)
# print(pose_base)

robot_ip = "192.168.0.32"
rtde_lock = multiprocessing.Lock()

rtde_r, rtde_c = connect_rtde(robot_ip)
if rtde_r is None or rtde_c is None:
    print("❌ Unable to establish RTDE connection. Exiting...")

    exit(1)

with rtde_lock:
    tcp_pose = rtde_r.getActualTCPPose()
    # tcp_vel_base = rtde_r.getActualTCPSpeed()

pose_base = ft.transform_pose(pose_in_tcp, from_frame="tcp", to_frame="base", pose_flange_base=tcp_pose)
print(pose_base)
# tcp_vel_base = ft.transform_velocity(velocity, tcp_pose_base, from_frame, to_frame, pose_flange_base=None)
print(tcp_pose)#, '\n', tcp_vel_base)