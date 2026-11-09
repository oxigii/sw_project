"""Background worker that runs the speaker/PTZ logic for Streamlit."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Event, Lock, Thread

import cv2
import numpy as np
from ultralytics import YOLO

from PTZ_control import yolo_rtsp_mic_detection as detection
from PTZ_control.visca_ptz_control import ViscaPTZController

MODE_IDLE, MODE_SINGLE, MODE_GROUP = "IDLE", "SINGLE", "GROUP"


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
    target_window_size: int = 3
    dead_zone_x: int = 120
    dead_zone_y: int = 90
    ptz_speed: int = 3


class PTZTrackingWorker:
    """Uses the standalone detector's tracking, smoothing and auto-zoom policy."""

    def __init__(self, config):
        self.config = config
        self._stop_event, self._state_lock, self._controller_lock = Event(), Lock(), Lock()
        self._thread = self._controller = self._capture = self._latest_frame = None
        self._logs, self._tracking_enabled = deque(maxlen=12), True
        self._state = {"running": False, "status": "대기", "camera_connected": False,
                       "source_mode": config.source_mode, "mode": MODE_IDLE, "mic_count": 0,
                       "person_count": 0, "fps": 0.0, "error": None}

    def start(self):
        if self._thread is not None and self._thread.is_alive(): return
        self._stop_event.clear(); self._set_state(running=True, status="시작 중", error=None)
        self._thread = Thread(target=self._run, name="ptz-tracking-worker", daemon=True); self._thread.start()

    def stop(self, timeout=3.0):
        self._stop_event.set(); self._release_capture()
        if self._thread is not None and self._thread.is_alive(): self._thread.join(timeout=timeout)

    def clear_screen(self):
        with self._state_lock:
            self._latest_frame = None; self._logs.clear()
            self._state.update(status="대기", camera_connected=False, mode=MODE_IDLE,
                               mic_count=0, person_count=0, fps=0.0, error=None)

    def set_tracking_enabled(self, enabled):
        with self._state_lock: self._tracking_enabled = bool(enabled)
        self._log("자동 추적 ON" if enabled else "자동 추적 OFF")

    def snapshot(self):
        with self._state_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            state = dict(self._state); state["tracking_enabled"] = self._tracking_enabled; state["logs"] = list(self._logs)
        return frame, state

    def manual_control(self, command):
        with self._state_lock:
            if self.config.source_mode != "ptz": return False, "노트북 캠 모드에서는 물리 PTZ 제어를 사용할 수 없습니다."
            if not self._state["camera_connected"]: return False, "카메라가 연결되어 있지 않습니다."
            if self._tracking_enabled: return False, "자동 추적을 끈 후 수동 제어해 주세요."
        try:
            with self._controller_lock:
                if self._controller is None: return False, "PTZ 제어 연결이 종료되었습니다."
                if command in {"left", "right", "up", "down"}: self._controller.move_manual(command)
                elif command == "zoom_in": self._controller.zoom_in()
                elif command == "zoom_out": self._controller.zoom_out(duration=0.25)
                elif command == "reset_view": self._controller.reset_view()
                else: return False, f"지원하지 않는 명령입니다: {command}"
            self._log(f"수동 명령: {command}"); return True, "카메라 명령을 전송했습니다."
        except (OSError, RuntimeError, ValueError) as exc:
            self._log(f"수동 제어 실패: {exc}"); return False, str(exc)

    def _set_state(self, **updates):
        with self._state_lock: self._state.update(updates)

    def _log(self, message):
        with self._state_lock: self._logs.appendleft(f"[{time.strftime('%H:%M:%S')}] {message}")

    def _release_capture(self):
        capture, self._capture = self._capture, None
        if capture is None: return
        try:
            if hasattr(capture, "stop"): capture.stop()
            else: capture.release()
        except cv2.error: pass

    def _open_capture(self):
        if self.config.source_mode == "ptz": return detection.LatestFrameCamera(self.config.rtsp_url)
        capture = cv2.VideoCapture(self.config.webcam_index); capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened(): capture.release(); raise RuntimeError("노트북 캠 영상을 열 수 없습니다.")
        return capture

    @staticmethod
    def _read_frame(capture):
        if isinstance(capture, detection.LatestFrameCamera): return capture.read()
        success, frame = capture.read(); return frame if success else None

    def _run(self):
        target_window = deque(maxlen=max(1, self.config.target_window_size))
        mic_hits, mic_scores = defaultdict(int), defaultdict(float)
        mic_search_active = False; mic_search_started = mic_pause_started = None; last_mic_detected = 0.0
        current_speaker_id = None; last_speaker_seen = 0.0; previous_people_count = 0
        camera_moving = False; zoom_state = None; frame_counter = 0; fps_started_at = time.monotonic()
        self._set_state(status="모델 불러오는 중", error=None); self._log("시스템 시작")
        try:
            detection.PERSON_CONF, detection.MIC_CONF = self.config.person_conf, self.config.mic_conf
            person_model, mic_model = YOLO("yolov8n.pt"), YOLO(self.config.mic_model_path)
            if self._stop_event.is_set(): return
            self._set_state(status="카메라 연결 중"); capture = self._open_capture(); self._capture = capture
            controller = None
            if self.config.source_mode == "ptz":
                controller = ViscaPTZController(camera_ip=self.config.camera_ip, port=self.config.visca_port,
                    speed=self.config.ptz_speed, dead_zone_x=self.config.dead_zone_x, dead_zone_y=self.config.dead_zone_y)
                controller.connect(); self._controller = controller
            self._set_state(status="실행 중", camera_connected=True)
            self._log("RTSP 및 VISCA 연결 완료" if controller else "노트북 캠 연결 완료")

            while not self._stop_event.is_set():
                frame = self._read_frame(capture)
                if frame is None: time.sleep(0.01); continue
                raw_frame, now = frame.copy(), time.monotonic()
                people = detection.detect_people(frame, person_model); people_count = len(people)
                target = None; mic_detections = []
                with self._state_lock: tracking_enabled = self._tracking_enabled

                # Same automatic zoom transitions as yolo_rtsp_mic_detection.py.
                requested_zoom = "SINGLE" if people_count == 1 else "GROUP" if people_count >= 2 else None
                if tracking_enabled and controller is not None and requested_zoom != zoom_state:
                    if requested_zoom is not None:
                        with self._controller_lock:
                            (controller.zoom_in if requested_zoom == "SINGLE" else controller.zoom_out)()
                        self._log("자동 줌인" if requested_zoom == "SINGLE" else "자동 줌아웃")
                    zoom_state = requested_zoom

                if people_count == 0:
                    current_speaker_id = None; target_window.clear(); mic_search_active = False
                    mic_search_started = mic_pause_started = None; mic_hits.clear(); mic_scores.clear()
                elif people_count == 1:
                    current_speaker_id = next(iter(people)); last_speaker_seen = now
                    target = detection.person_target(people[current_speaker_id])
                    mic_search_active = False; mic_search_started = mic_pause_started = None; mic_hits.clear(); mic_scores.clear()
                else:
                    if previous_people_count < 2 or current_speaker_id is None:
                        if not mic_search_active:
                            mic_search_active, mic_search_started, mic_pause_started = True, now, None; mic_hits.clear(); mic_scores.clear(); self._log("다중 인원: 마이크 탐색 시작")
                    if current_speaker_id in people: last_speaker_seen = now
                    elif current_speaker_id is not None and now - last_speaker_seen > detection.SPEAKER_LOST_TIMEOUT:
                        current_speaker_id = None; mic_search_active, mic_search_started, mic_pause_started = True, now, None; mic_hits.clear(); mic_scores.clear()
                    if mic_search_active:
                        if now - last_mic_detected >= detection.MIC_DETECT_INTERVAL:
                            mic_detections = detection.detect_microphones_for_people(frame, people, mic_model); last_mic_detected = now
                            for item in mic_detections: mic_hits[item["track_id"]] += 1; mic_scores[item["track_id"]] += item["confidence"]
                        if now - mic_search_started >= detection.MIC_SEARCH_DURATION:
                            current_speaker_id = detection.choose_speaker(current_speaker_id, mic_hits, mic_scores)
                            mic_search_active, mic_search_started, mic_pause_started = False, None, now; mic_hits.clear(); mic_scores.clear(); self._log("마이크 탐색 완료")
                    elif mic_pause_started is None or now - mic_pause_started >= detection.MIC_SEARCH_PAUSE:
                        mic_search_active, mic_search_started, mic_pause_started = True, now, None; mic_hits.clear(); mic_scores.clear()
                    if current_speaker_id in people: target = detection.person_target(people[current_speaker_id])

                if tracking_enabled and controller is not None:
                    if target is None:
                        target_window.clear()
                        if camera_moving:
                            with self._controller_lock: controller.pan_tilt_stop()
                            camera_moving = False
                    else:
                        target_window.append(target); smooth_target = np.mean(target_window, axis=0); height, width = frame.shape[:2]
                        if detection.is_target_in_stop_zone(smooth_target, width, height):
                            if camera_moving:
                                with self._controller_lock: controller.pan_tilt_stop()
                                camera_moving = False
                        else:
                            speed = detection.SLOW_SPEED if detection.is_target_in_center_zone(smooth_target, width, height) else detection.FAST_SPEED
                            with self._controller_lock: camera_moving = controller.move_to_target(*smooth_target, width, height, speed=speed) or camera_moving
                elif not tracking_enabled and camera_moving and controller is not None:
                    with self._controller_lock: controller.pan_tilt_stop()
                    camera_moving = False

                mode = MODE_IDLE if people_count == 0 else MODE_SINGLE if people_count == 1 else MODE_GROUP
                frame_counter += 1; elapsed = time.monotonic() - fps_started_at
                fps = frame_counter / elapsed if elapsed >= 1 else self.snapshot()[1]["fps"]
                if elapsed >= 1: frame_counter, fps_started_at = 0, time.monotonic()
                with self._state_lock:
                    self._latest_frame = raw_frame
                    self._state.update(status="실행 중", mode=mode, mic_count=len(mic_detections), person_count=people_count, fps=fps)
                previous_people_count = people_count
        except Exception as exc:
            self._set_state(status="오류", error=str(exc)); self._log(f"오류: {exc}")
        finally:
            self._set_state(running=False, camera_connected=False); self._release_capture()
            if self._controller is not None:
                try:
                    with self._controller_lock: self._controller.close()
                except OSError: pass
                self._controller = None
            if self._state["status"] != "오류": self._set_state(status="중지됨")
            self._log("시스템 종료")
