from RTDE_utils import RTDEManager
from frame_transformer import FrameTransformer
from AdmittanceControl_utils import AdmittanceControl

import keyboard
import traceback
import multiprocessing
import numpy as np

def skew(v):
    """Returns the skew-symmetric matrix of a 3D vector."""
    v = np.asarray(v).flatten()  # Ensure v is a 1D NumPy array
    return np.array([
        [  0,   -v[2],  v[1]],
        [ v[2],   0,   -v[0]],
        [-v[1],  v[0],   0]
    ])

def apply_threshold(value):
    max_value = 5
    result = value - 5.0 if value > 5.0 else value + 5.0 if value < -5.0 else 0.0
    return max(-max_value, min(result, max_value))

def apply_torque_threshold(value):
    return value - 1 if value > 1 else value + 1 if value < -1 else 0.0

# === 機器人與感測器參數設定 ===
robot_ip = "192.168.0.32"

M_t = [11.2635, 11.2635, 5.0] # Mass matrix for translation (kg)
K_t = [181.4451, 181.4451, 100.4451] # Stiffness matrix for translation (N/m)

M_r = [10.0, 10.0, 10.0] # Inertia matrix for rotation
K_r = [20.0, 20.0, 20.0] # Stiffness matrix for rotation

offset_base_world = [0, 0, 0, 0, 0, 0]
offset_tcp_flange  = [0, 0, 0, 0, 0, 0]

rtde_lock = multiprocessing.Lock()

rtde_mgr = RTDEManager(robot_ip, use_receive=True,
                       use_control=True, use_io=False)

ad_c = AdmittanceControl(M_trans = M_t, K_trans = None, Zeta_trans = None,
                 M_rot = M_r, K_rot = None, Zeta_rot = None)

transformer = FrameTransformer(offset_base_world, offset_tcp_flange)

if not rtde_mgr.check_and_reconnect():
    print("❌ Unable to establish RTDE connection. Exiting...")
    exit(1)

rtde_r = rtde_mgr.get_interface('receive')
rtde_c = rtde_mgr.get_interface('control')

tcp_pose = rtde_r.getActualTCPPose()
x_init, r_init = transformer.get_pos_rotm(tcp_pose)
tcp_force_init = np.array(rtde_r.getActualTCPForce())
v_current = np.array([[0.0], [0.0], [0.0]])
v_des = np.array([[0.0], [0.0], [0.0]])
a_current = np.array([[0.0], [0.0], [0.0]])
w_current = np.array([[0.0], [0.0], [0.0]])
w_des = np.array([[0.0], [0.0], [0.0]])
alpha_current = np.array([[0.0], [0.0], [0.0]]) 

print(x_init, r_init)
print(tcp_force_init)

velocity = 3
acceleration = 3
dt = 1.0/500
lookahead_time = 0.1
gain = 300

x_current, r_current = transformer.get_pos_rotm(tcp_pose)
try:
    while True:
        t_start = rtde_c.initPeriod()

        with rtde_lock:
            if rtde_mgr.check_and_reconnect():
                pass
            else:
                print("❌ Lost RTDE connection. Exiting loop...")
                break
            
            tcp_force = np.array(rtde_r.getActualTCPForce())
            tcp_pose = rtde_r.getActualTCPPose()
            x_current, r_mes = transformer.get_pos_rotm(tcp_pose)

        force = np.array([[0], [0], [apply_threshold(tcp_force[2])]])
        # force = np.array([[0], [0], [0]])
        torque = np.array([[0], [0], [tcp_force[5]]])


        a_current = ad_c.translational(force, np.array(x_init).reshape(3, 1), np.array(x_current).reshape(3, 1), v_des, v_current)
        v_current, x_command = ad_c.trans_integrate(np.array(x_current).reshape(3, 1), v_current, a_current, dt, mode="position")

        alpha_current = ad_c.rotational(torque, r_init, r_mes, w_des, w_current)
        w_current, r_current, rotation_vector = ad_c.rot_integrate(r_init, r_current, w_current, alpha_current, dt, mode="position")
        
        pose = np.array([x_command.T[0][0], x_command.T[0][1], x_command.T[0][2],
                    rotation_vector[0], rotation_vector[1], rotation_vector[2]])

        rtde_c.servoL(pose.tolist(), velocity, acceleration, dt, lookahead_time, gain)
        rtde_c.waitPeriod(t_start)
        if keyboard.is_pressed("esc"):
            print("\n🛑 'Esc' pressed, exiting...")
            break
except Exception as e:
    print("❌ Unexpected error occurred:")
    traceback.print_exc()

finally:
    rtde_mgr.disconnect()