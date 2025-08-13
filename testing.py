from RTDE_utils import RTDEManager
from frame_transformer import FrameTransformer
from FT300_utils import ForceSensor
from Plot_FT import live_plot_ft_data

import time
import keyboard
import traceback
import multiprocessing
import numpy as np
from functools import partial

# === Robot and Sensor Configuration ===
robot_ip = "192.168.0.32"
port_name = "COM5"  # Serial port for force/torque sensor

offset_base_world = [0, 0, 0, 0, 0, 0]
offset_tcp_flange  = [0, 0, 0, 0, 0, 0]
control_freq = 100  # Hz
dt = 1.0 / control_freq

# Initialize transformer for frame conversions
transformer = FrameTransformer(offset_base_world, offset_tcp_flange)

# Multiprocessing shared data
FT_data = multiprocessing.Array('d', 6)
ft_lock = multiprocessing.Lock()
rtde_lock = multiprocessing.Lock()
wrench_world_shared = multiprocessing.Array('d', 6)

def get_latest_wrench(wrench_world_shared):
    return np.array(wrench_world_shared[:])
get_wrench_func = partial(get_latest_wrench, wrench_world_shared)

# Initialize RTDE manager outside main (for safer process startup on Windows)
rtde_mgr = RTDEManager(robot_ip, frequency = 100, use_receive=True, use_control=False, use_io=False)

if __name__ == "__main__":
    # Start real-time force/torque plotting process
    plot_process = multiprocessing.Process(
        target=live_plot_ft_data,
        args=(get_wrench_func,),
        kwargs={"interval": 0.002, "max_len": 200, "title": "Wrench in World Frame"}
    )
    plot_process.start()

    # Start force sensor reading process
    sensor = ForceSensor(port_name=port_name)
    sensor_process = multiprocessing.Process(
        target=sensor.read_sensor, 
        args=(FT_data, ft_lock), 
        kwargs={"use_zero_ref": True}
    )
    sensor_process.daemon = True
    sensor_process.start()

    # Get RTDE interface handlers
    rtde_r = rtde_mgr.get_interface('receive')
    if rtde_r is None:
        print("❌ Unable to establish RTDE connection. Exiting...")
        sensor_process.terminate()
        plot_process.terminate()
        sensor_process.join()
        plot_process.join()
        exit(1)

    print('Waiting for force sensor to initialize. Please hold on.')
    for i in range(5, 0, -1):
        print(f'Starting in {i} seconds...', end='\r')
        time.sleep(1)
    print('Start!')

    try:
        next_time = time.perf_counter()
        count = 0
        last_print = time.perf_counter()

        while True:
            # --- Robot state update with RTDE ---
            with rtde_lock:
                if not rtde_r.isConnected():
                    print("❌ Lost RTDE connection. Exiting loop...")
                    break
                tcp_pose_base = rtde_r.getActualTCPPose()
                tcp_vel_base = rtde_r.getActualTCPSpeed()

            # --- Force sensor data update ---
            with ft_lock:
                wrench_flange = np.array(FT_data)
            
            # --- Transform wrench from flange frame to world frame ---
            wrench_world = transformer.transform_wrench(
                wrench_flange,
                tcp_pose_base=None,
                from_frame="flange",
                to_frame="world",
                pose_flange_base=tcp_pose_base
            )

            # Write to shared memory for plotting process
            wrench_world_shared[:] = wrench_flange

            count += 1

            # Print loop frequency every second
            now = time.perf_counter()
            if now - last_print >= 1.0:
                print(f"Control loop frequency: {count} Hz")
                count = 0
                last_print += 1.0

            # --- Period control for real-time loop ---
            next_time += dt
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                if sleep_time > 0.002:
                    time.sleep(sleep_time - 0.001)
                while time.perf_counter() < next_time:
                    pass
            else:
                next_time = time.perf_counter()

            if keyboard.is_pressed("esc"):
                print("\n🛑 'Esc' pressed, program will exit.")
                break

    except Exception as e:
        print("❌ Unexpected error occurred:")
        traceback.print_exc()

    finally:
        rtde_mgr.disconnect()
        sensor_process.terminate()
        plot_process.terminate()
        sensor_process.join()
        plot_process.join()
