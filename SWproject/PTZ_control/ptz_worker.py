"""Background video-processing worker used by the Streamlit UI."""

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from PTZ_control import yolo_rtsp_mic_detection as detection
from PTZ_control.visca_ptz_control import ViscaPTZController


@dataclass(frozen=True)
class WorkerConfig:
    source_mode: str
    rtsp_url: str
    mic_model_path: str
    camera_ip: str
    webcam_index: int = 0
    digital_zoom: float = 1.6
    visca_port: int = 5678
    person_conf: float = 0.5
    mic_conf: float = 0.5
    mode_confirm_frames: int = 10
    target_window_size: int = 5
    dead_zone_x: int = 120
    dead_zone_y: int = 90
    ptz_speed: int = 3


class PTZTrackingWorker:
    """Read RTSP frames and run YOLO/PTZ control outside Streamlit's UI thread."""

    def __init__(self, config):
        self.config = config
        self._stop_event = threading.Event()
        self._thread = None
        self._state_lock = threading.Lock()
        self._controller_lock = threading.Lock()
        self._controller = None
        self._capture = None
        self._latest_frame = None
        self._logs = deque(maxlen=12)
        self._tracking_enabled = True
        self._state = {
            "running": False,
            "status": "대기",
            "camera_connected": False,
            "source_mode": self.config.source_mode,
            "mode": detection.MODE_IDLE,
            "mic_count": 0,
            "person_count": 0,
            "fps": 0.0,
            "error": None,
        }

    @property
    def running(self):
        with self._state_lock:
            return bool(self._state["running"])

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._set_state(running=True, status="시작 중", error=None)
        self._thread = threading.Thread(
            target=self._run,
            name="ptz-tracking-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout=3.0):
        self._stop_event.set()

        capture = self._capture
        if capture is not None:
            capture.release()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def clear_screen(self):
        with self._state_lock:
            self._latest_frame = None
            self._logs.clear()
            self._state.update(
                {
                    "status": "대기",
                    "camera_connected": False,
                    "mode": detection.MODE_IDLE,
                    "mic_count": 0,
                    "person_count": 0,
                    "fps": 0.0,
                    "error": None,
                }
            )

    def set_tracking_enabled(self, enabled):
        with self._state_lock:
            self._tracking_enabled = bool(enabled)
        self._log("자동 추적 ON" if enabled else "자동 추적 OFF")

    def snapshot(self):
        with self._state_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            state = dict(self._state)
            state["tracking_enabled"] = self._tracking_enabled
            state["logs"] = list(self._logs)
        return frame, state

    def manual_control(self, command):
        """Execute a manual PTZ command while automatic tracking is disabled."""
        with self._state_lock:
            if self.config.source_mode != "ptz":
                return False, "노트북 캠 모드에서는 물리 PTZ 제어를 사용할 수 없습니다."
            if not self._state["camera_connected"]:
                return False, "카메라가 연결되어 있지 않습니다."
            if self._tracking_enabled:
                return False, "자동 추적을 끈 후 수동 제어해 주세요."

        try:
            with self._controller_lock:
                if command in {"left", "right", "up", "down"}:
                    self._controller.move_manual(command)
                elif command == "zoom_in":
                    self._controller.zoom_in()
                elif command == "zoom_out":
                    self._controller.zoom_out(duration=0.25)
                elif command == "reset_view":
                    self._controller.reset_view()
                else:
                    return False, f"지원하지 않는 명령입니다: {command}"
            self._log(f"수동 명령: {command}")
            return True, "카메라 명령을 전송했습니다."
        except (OSError, RuntimeError, ValueError) as exc:
            self._log(f"수동 제어 실패: {exc}")
            return False, str(exc)

    def _set_state(self, **updates):
        with self._state_lock:
            self._state.update(updates)

    def _log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        with self._state_lock:
            self._logs.appendleft(f"[{timestamp}] {message}")

    @staticmethod
    def _digital_follow(frame, target, mode, zoom_factor):
        """Simulate PTZ by cropping around one target and resizing the frame."""
        if mode != detection.MODE_SINGLE or target is None or zoom_factor <= 1.0:
            return frame

        frame_height, frame_width = frame.shape[:2]
        crop_width = max(1, int(frame_width / zoom_factor))
        crop_height = max(1, int(frame_height / zoom_factor))
        target_x, target_y = map(float, target)

        left = int(np.clip(target_x - crop_width / 2, 0, frame_width - crop_width))
        top = int(np.clip(target_y - crop_height / 2, 0, frame_height - crop_height))
        cropped = frame[top : top + crop_height, left : left + crop_width]

        return cv2.resize(
            cropped,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _run(self):
        mode_manager = detection.CameraModeManager(
            confirm_frames=self.config.mode_confirm_frames
        )
        target_window = deque(maxlen=self.config.target_window_size)
        group_framed = False
        last_single_target = None
        missing_target_frames = 0
        frame_counter = 0
        fps_started_at = time.monotonic()

        self._set_state(status="모델 불러오는 중", error=None)
        self._log("시스템 시작")

        try:
            detection.PERSON_CONF = self.config.person_conf
            detection.MIC_CONF = self.config.mic_conf

            person_model = YOLO("yolov8n.pt")
            mic_model = YOLO(self.config.mic_model_path)

            if self._stop_event.is_set():
                return

            is_webcam = self.config.source_mode == "webcam"
            source = self.config.webcam_index if is_webcam else self.config.rtsp_url
            source_name = "노트북 캠" if is_webcam else "RTSP"

            self._set_state(status=f"{source_name} 연결 중")
            capture = cv2.VideoCapture(source)
            self._capture = capture
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not capture.isOpened():
                raise RuntimeError(f"{source_name} 영상을 열 수 없습니다.")
            if self._stop_event.is_set():
                return

            controller = None
            if not is_webcam:
                controller = ViscaPTZController(
                    camera_ip=self.config.camera_ip,
                    port=self.config.visca_port,
                    speed=self.config.ptz_speed,
                    dead_zone_x=self.config.dead_zone_x,
                    dead_zone_y=self.config.dead_zone_y,
                )
                controller.connect()
                self._controller = controller

            self._set_state(status="실행 중", camera_connected=True)
            if is_webcam:
                self._log("노트북 캠 연결 완료 - 디지털 추적 모드")
            else:
                self._log("RTSP 및 VISCA 연결 완료")

            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError("RTSP 프레임 수신에 실패했습니다.")

                person_boxes, mics = detection.detect_mics(
                    frame,
                    person_model,
                    mic_model,
                )
                mode, mode_changed = mode_manager.update(len(mics))

                if mode_changed:
                    target_window.clear()
                    group_framed = False
                    if mode != detection.MODE_SINGLE:
                        last_single_target = None
                        missing_target_frames = 0
                    self._log(f"추적 모드 변경: {mode}")

                with self._state_lock:
                    tracking_enabled = self._tracking_enabled

                target = detection.calculate_target(mics, mode)
                if target is not None:
                    target_window.append(target)
                    target = np.mean(target_window, axis=0)
                    if mode == detection.MODE_SINGLE:
                        last_single_target = target.copy()
                        missing_target_frames = 0
                elif (
                    mode == detection.MODE_SINGLE
                    and last_single_target is not None
                    and missing_target_frames < self.config.mode_confirm_frames
                ):
                    missing_target_frames += 1
                    target = last_single_target

                if (
                    tracking_enabled
                    and target is not None
                    and not is_webcam
                    and controller is not None
                ):
                    if mode == detection.MODE_GROUP and not group_framed:
                        with self._controller_lock:
                            controller.zoom_out()
                        group_framed = True
                        self._log("그룹 화면 전환: 줌아웃")
                    elif mode != detection.MODE_GROUP:
                        group_framed = False

                    frame_height, frame_width = frame.shape[:2]
                    with self._controller_lock:
                        controller.move_to_target(
                            float(target[0]),
                            float(target[1]),
                            frame_width,
                            frame_height,
                        )

                detection.draw_results(
                    frame,
                    person_boxes,
                    mics,
                    mode,
                    target,
                )

                display_frame = frame
                if is_webcam and tracking_enabled:
                    display_frame = self._digital_follow(
                        frame,
                        target,
                        mode,
                        self.config.digital_zoom,
                    )

                frame_counter += 1
                elapsed = time.monotonic() - fps_started_at
                if elapsed >= 1.0:
                    fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_started_at = time.monotonic()
                else:
                    with self._state_lock:
                        fps = self._state["fps"]

                with self._state_lock:
                    self._latest_frame = display_frame
                    self._state.update(
                        {
                            "status": "실행 중",
                            "mode": mode,
                            "mic_count": len(mics),
                            "person_count": len(person_boxes),
                            "fps": fps,
                        }
                    )

        except Exception as exc:
            self._set_state(status="오류", error=str(exc))
            self._log(f"오류: {exc}")
        finally:
            self._set_state(running=False, camera_connected=False)

            capture = self._capture
            self._capture = None
            if capture is not None:
                capture.release()

            controller = self._controller
            self._controller = None
            if controller is not None:
                try:
                    with self._controller_lock:
                        controller.close()
                except OSError:
                    pass

            if self._state["status"] != "오류":
                self._set_state(status="중지됨")
            self._log("시스템 종료")
