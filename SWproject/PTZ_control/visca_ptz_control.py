"""VISCA over IP PTZ control helpers.

This module is imported by ``yolo_rtsp_mic_detection.py``. It does not move
the camera when imported; the connection and movement are managed through the
``ViscaPTZController`` class.
"""

import os
import socket
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
CAMERA_IP = os.getenv("PTZ_CAMERA_IP", "")
VISCA_PORT = 5678

DEAD_ZONE_X = 120
DEAD_ZONE_Y = 90
PTZ_SPEED = 6
PTZ_SLOW_SPEED = 2
PTZ_MOVE_TIME = 0.08
PTZ_COMMAND_INTERVAL = 0.15
GROUP_ZOOM_OUT_TIME = 0.8


class ViscaPTZController:
    def __init__(
        self,
        camera_ip=CAMERA_IP,
        port=VISCA_PORT,
        speed=PTZ_SPEED,
        move_time=PTZ_MOVE_TIME,
        command_interval=PTZ_COMMAND_INTERVAL,
        dead_zone_x=DEAD_ZONE_X,
        dead_zone_y=DEAD_ZONE_Y,
    ):
        self.camera_ip = camera_ip
        self.port = port
        self.speed = speed
        self.move_time = move_time
        self.command_interval = command_interval
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y
        self.sock = None
        self.last_command_time = 0.0

    def connect(self):
        if self.sock is not None:
            return

        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.settimeout(5)
        try:
            candidate.connect((self.camera_ip, self.port))
        except OSError:
            candidate.close()
            raise

        self.sock = candidate
        print(f"[VISCA] connected: {self.camera_ip}:{self.port}")

    def close(self):
        if self.sock is None:
            return

        try:
            try:
                self.pan_tilt_stop()
                self.zoom_stop()
            except OSError:
                pass
        finally:
            self.sock.close()
            self.sock = None

    def send_visca(self, hex_cmd):
        if self.sock is None:
            raise RuntimeError("VISCA socket is not connected")

        self.sock.sendall(bytes.fromhex(hex_cmd))

    def pan_tilt_stop(self):
        self.send_visca("81 01 06 01 00 00 03 03 FF")

    def zoom_stop(self):
        self.send_visca("81 01 04 07 00 FF")

    def zoom_in(self, duration=0.25):
        """Zoom toward telephoto for a short time, then stop."""
        self.send_visca("81 01 04 07 02 FF")
        time.sleep(duration)
        self.zoom_stop()

    def zoom_out(self, duration=GROUP_ZOOM_OUT_TIME):
        """Zoom toward wide angle for a short time, then stop."""
        self.send_visca("81 01 04 07 03 FF")
        time.sleep(duration)
        self.zoom_stop()

    def move_manual(self, direction, duration=1.5):
        """Move in one direction for manual UI control."""
        directions = {
            "left": ("01", "03"),
            "right": ("02", "03"),
            "up": ("03", "01"),
            "down": ("03", "02"),
        }
        if direction not in directions:
            raise ValueError(f"Unsupported PTZ direction: {direction}")

        pan_direction, tilt_direction = directions[direction]
        speed_hex = f"{self.speed:02X}"
        self.send_visca(
            f"81 01 06 01 {speed_hex} {speed_hex} "
            f"{pan_direction} {tilt_direction} FF"
        )
        time.sleep(duration)
        self.pan_tilt_stop()

    def reset_view(self, zoom_out_time=2.0):
        """Return pan/tilt home and widen the view for a full-room shot."""
        self.send_visca("81 01 06 04 FF")
        time.sleep(0.8)
        self.zoom_out(duration=zoom_out_time)

    def move_to_target(self, target_x, target_y, frame_width, frame_height, speed=None):
        """Move PTZ toward target. Stop-zone logic is handled externally."""

        now = time.monotonic()

        if now - self.last_command_time < self.command_interval:
            return False

        # 화면 중앙
        center_x = frame_width / 2
        center_y = frame_height / 2

        # 타겟과 화면 중앙의 차이
        error_x = target_x - center_x
        error_y = (target_y - center_y) * 0.15 #y축 움직이는 범위?

        # 사용할 속도
        if speed is None:
            speed = self.speed

        pan_speed = speed
        tilt_speed = max(1, int(speed * 0.5))

        pan_speed_hex = f"{pan_speed:02X}"
        tilt_speed_hex = f"{tilt_speed:02X}"

        # -----------------------------------------
        # 방향 결정
        # -----------------------------------------

        # X 방향
        if error_x < 0:
            pan_direction = "01"      # left
        elif error_x > 0:
            pan_direction = "02"      # right
        else:
            pan_direction = "03"      # stop

        # Y 방향
        if error_y < 0:
            tilt_direction = "01"     # up
        elif error_y > 0:
            tilt_direction = "02"     # down
        else:
            tilt_direction = "03"     # stop

        self.send_visca(
            f"81 01 06 01 "
            f"{pan_speed_hex} {tilt_speed_hex} "
            f"{pan_direction} {tilt_direction} FF"
        )

        self.last_command_time = now

        return True 

if __name__ == "__main__":
    controller = ViscaPTZController()
    try:
        controller.connect()
        print("left movement test")
        controller.send_visca("81 01 06 01 05 05 01 03 FF")
        time.sleep(1)
        controller.pan_tilt_stop()
        print("test completed")
    except Exception as exc:
        print("VISCA control failed:", exc)
    finally:
        controller.close()
