"""
RTDEManager Class for Robust, Flexible UR RTDE Connection Management

This class provides unified management for Universal Robots RTDE communication (rtde_receive, rtde_control, rtde_io).
Users can flexibly select which interfaces to connect. The class supports automatic connection, reconnection,
and safe disconnection. It is suitable for applications requiring multi-interface UR RTDE API operations.

Documentation:
--------------
For further details, see the official introduction and API:
https://sdurobotics.gitlab.io/ur_rtde/introduction/introduction.html

Features:
---------
- User-defined selection of RTDE interfaces to connect (Receive/Control/IO, any combination)
- Automatic interface connection and reconnection (receive/control)
- Unified safe disconnection of all interfaces

Dependencies:
-------------
Requires ur_rtde version 1.6.0
- rtde_receive
- rtde_control
- rtde_io
- time

Note:
-----
- Please ensure you are using ur_rtde version 1.6.0.
- IO interface (RTDEIOInterface) is only initialized once and not automatically checked/reconnected.
  Use try/except for IO error handling in your application logic.
- This class only manages connection/disconnection/status, not robot motion or data operations.
- Refer to the official UR RTDE API for further usage.

Author: Yu-Peng, Yeh, 2025-07-09, nickyeh0415@gmail.com
"""

import time
import rtde_receive
import rtde_control
import rtde_io

class RTDEManager:
    """
    RTDEManager provides unified connection management for UR robots.
    Allows flexible combination of RTDE interfaces with robust auto-reconnection and disconnection.
    """

    def __init__(self, robot_ip, frequency=500, retries=5, delay=2,
                 use_receive=True, use_control=False, use_io=False):
        """
        Initialize RTDEManager and connect the specified RTDE interfaces.

        Args:
            robot_ip (str): UR robot IP address.
            frequency (int): Data communication frequency (Hz), default 500, for UR10e is 500 Hz, UR5 is 125 Hz.
            retries (int): Maximum retry attempts for connection, default 5.
            delay (float): Delay (sec) between retries, default 2.
            use_receive (bool): Enable RTDEReceiveInterface.
            use_control (bool): Enable RTDEControlInterface.
            use_io (bool): Enable RTDEIOInterface.
        """
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.retries = retries
        self.delay = delay

        self.use_receive = use_receive
        self.use_control = use_control
        self.use_io = use_io

        self.rtde_r = None
        self.rtde_c = None
        self.rtde_io = None
        self.connect()

    def connect(self):
        """
        Establish connection(s) to the enabled RTDE interfaces.

        Returns:
            bool: True if all enabled interfaces connected successfully, else False.
        """
        for attempt in range(self.retries):
            try:
                if self.use_receive:
                    self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip, self.frequency)
                if self.use_control:
                    self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
                if self.use_io and self.rtde_io is None:
                    self.rtde_io = rtde_io.RTDEIOInterface(self.robot_ip)
                print("✅ RTDE Interface(s) Connected")
                return True
            except Exception as e:
                print(f"⚠️ RTDE Connection Attempt {attempt+1} Failed: {e}")
                self.disconnect()  # Ensure all interface are released before retry
                time.sleep(self.delay)
        print("❌ RTDE Connection Failed after multiple attempts.")
        return False

    def disconnect(self):
        """
        Safely disconnect all enabled RTDE interfaces and release resources.
        """
        if self.rtde_r:
            try:
                self.rtde_r.disconnect()
                print("🛑 RTDEReceiveInterface disconnected")
            except Exception:
                pass
            self.rtde_r = None
        if self.rtde_c:
            try:
                self.rtde_c.disconnect()
                print("🛑 RTDEControlInterface disconnected")
            except Exception:
                pass
            self.rtde_c = None
        if self.rtde_io:
            try:
                self.rtde_io.disconnect()
                print("🛑 RTDEIOInterface disconnected")
            except Exception:
                pass
            self.rtde_io = None

    def check_and_reconnect(self):
        """
        Check connection status of enabled interfaces (receive/control only).
        Reconnect automatically if any connection is lost.

        Returns:
            bool: True if all enabled interfaces are connected, else False.
        """
        reconnect_required = False

        if self.use_receive and (self.rtde_r is None or not self.rtde_r.isConnected()):
            print("🔄 RTDE Receive connection lost. Attempting to reconnect...")
            reconnect_required = True

        if self.use_control and (self.rtde_c is None or not self.rtde_c.isConnected()):
            print("🔄 RTDE Control connection lost. Attempting to reconnect...")
            reconnect_required = True

        if reconnect_required:
            self.disconnect()
            if self.connect():
                print("✅ RTDE Interface(s) Reconnected")
                return True
            else:
                print("❌ RTDE reconnection failed.")
                return False

        return True

    def get_interface(self, which='receive'):
        """
        Get the handler for the specified RTDE interface.

        Args:
            which (str): One of 'receive', 'control', or 'io'.

        Returns:
            RTDE interface object or None.

        Raises:
            ValueError: If argument is not 'receive', 'control', or 'io'.
        """
        if which == 'receive':
            return self.rtde_r
        elif which == 'control':
            return self.rtde_c
        elif which == 'io':
            return self.rtde_io
        else:
            raise ValueError("Argument 'which' must be one of: 'receive', 'control', or 'io'.")

# Example usage:
if __name__ == "__main__":
    mgr = RTDEManager("192.168.0.32", use_receive=True, use_control=True, use_io=True)

    if mgr.check_and_reconnect():
        rtde_r = mgr.get_interface('receive')
        rtde_i = mgr.get_interface('io')
        rtde_c = mgr.get_interface('control')
        try:
            rtde_i.setAnalogOutputVoltage(0, 0)
            print("TCP position:", rtde_r.getActualTCPPose())
        except Exception as e:
            print("IO operation failed:", e)
    else:
        print("Failed to restore connection.")

    mgr.disconnect()

