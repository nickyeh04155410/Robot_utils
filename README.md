# Robot_utils

Modular utilities for real-time robotic control using Universal Robots (UR) RTDE interface and 6-DOF admittance control.  
This project supports Cartesian impedance/admittance control, force/torque sensor processing, and multi-frame coordinate transformations.

---

### 1. `RTDE_utils.py`
RTDE interface manager for robust, flexible connection handling.

**Features**:
- Unified management of `rtde_receive`, `rtde_control`, and `rtde_io`
- Automatic reconnection support
- Safe disconnect and resource cleanup
- Interface selector (`get_interface()`)

**Dependency**:
- Requires `ur_rtde==1.6.0`

For further details, see the official introduction and API:
https://sdurobotics.gitlab.io/ur_rtde/introduction/introduction.html

---

### 2. `frame_transformer.py`
Generic multi-frame 6D pose and vector transformer.

**Features**:
- Arbitrary tree-based frame structure (world → base → flange → tcp, + cameras)
- Transform pose, velocity, and wrench across frames
- Homogeneous transformation using Rodrigues vectors
- Supports dynamic links like `flange`

---

### 3. `AdmittanceControl_utils.py`
Core class for translational and rotational **admittance control**.

**Features**:
- Decoupled 3×3 diagonal admittance model for translation and rotation
- Force → acceleration conversion using virtual mass-damping-stiffness

**Note**:
- Force/torque expected in **base/world** frame

---

### 4. `test_admittance.py`
Main script to run Cartesian admittance control on a UR robot in real time.  
It connects to the UR via RTDE, reads TCP force/torque data, computes acceleration via admittance control, and commands new poses.

**Key Features**:
- Real-time translational and rotational admittance control
- Force filtering and saturation threshold
- Servo loop at 500 Hz

---
