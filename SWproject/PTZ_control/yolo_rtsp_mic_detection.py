"""RTSP speaker detection and automatic PTZ framing.

Control policy
--------------
0 people:
    Stop PTZ.

1 person:
    Follow that person directly.
    Microphone detection is not used.

2+ people:
    Use YOLO tracking IDs to maintain person identity.

    Microphone search cycle:
        Search for at least 3 seconds.
        If no reliable microphone is found, keep searching.
        After a reliable detection, pause 3 seconds and repeat.

    When there is no current speaker, microphone search starts immediately.

    Microphone detections are used ONLY to determine which person is
    the speaker. PTZ always follows the selected PERSON, not the
    microphone position.

PTZ movement:
    Outside center zone -> FAST
    Inside center zone  -> SLOW
    Inside stop zone    -> STOP
"""

import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO


# ============================================================
# RTSP CAMERA
# ============================================================

class LatestFrameCamera:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url

        self.cap = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        # 가능한 경우 RTSP 버퍼 최소화
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"RTSP camera connection failed: {rtsp_url}"
            )

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()

    def _reader(self):
        while self.running:
            success, frame = self.cap.read()

            if not success:
                time.sleep(0.01)
                continue

            # 항상 최신 프레임만 유지
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        self.running = False

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self.cap.release()


# ============================================================
# PTZ IMPORT
# ============================================================

try:
    from .visca_ptz_control import ViscaPTZController
except ImportError:
    from visca_ptz_control import ViscaPTZController


# ============================================================
# PATH / MODEL SETTINGS
# ============================================================

PTZ_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PTZ_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

RTSP_URL = os.getenv("PTZ_RTSP_URL", "")

MIC_MODEL_PATH = os.getenv(
    "MIC_MODEL_PATH",
    str(
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "train"
        / "weights"
        / "best.pt"
    ),
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

PERSON_CONF = 0.5

# 마이크 모델이 약하기 때문에 너무 높게 잡지 않음
MIC_CONF = 0.35

# 사람 ROI를 조금 확장
PADDING = 20


# ============================================================
# MICROPHONE SEARCH CYCLE
# ============================================================

# 3초 쉬고
MIC_SEARCH_PAUSE = 3.0

# 3초 동안 탐색
MIC_SEARCH_DURATION = 3.0

# 탐색 중 마이크 YOLO 실행 간격
# 0.2초 = 3초 동안 최대 약 15회 검사
MIC_DETECT_INTERVAL = 0.2


# ============================================================
# SPEAKER DECISION
# ============================================================

# 최소한 이 횟수 이상 검출되어야 발화자 후보로 인정
MIN_MIC_HITS = 3

# 기존 발화자를 변경할 때 필요한 추가 점수
SPEAKER_SWITCH_MARGIN = 2.0

# 사람 ID가 잠깐 사라졌을 때 바로 발화자를 버리지 않기 위한 시간
SPEAKER_LOST_TIMEOUT = 1.5


# ============================================================
# TARGET SMOOTHING
# ============================================================

TARGET_WINDOW_SIZE = 2


# ============================================================
# PTZ CENTER / STOP AREA
# ============================================================

# 큰 영역
# 이 안으로 들어오면 저속 이동
CENTER_ZONE_X = 0.20
CENTER_ZONE_Y = 0.20

# 작은 영역
# 여기까지 들어오면 완전 정지
STOP_ZONE_X = 0.12
STOP_ZONE_Y = 0.08

# PTZ가 한번 중앙에 맞은 뒤에는
# 이 범위를 벗어나기 전까지 다시 움직이지 않음
PTZ_UNLOCK_ZONE_X = 0.18
PTZ_UNLOCK_ZONE_Y = 0.12

# Runtime settings shared safely with the FastAPI worker.
class TrackingSettings:
    VALID_MODES = {"fixed", "professor", "multi"}
    VALID_POSITIONS = {"left", "center", "right"}

    def __init__(self):
        self._lock = threading.Lock()
        self._mode = "multi"
        self._position = "center"
        self._revision = 0

    def snapshot(self):
        with self._lock:
            return self._mode, self._position, self._revision

    def update(self, *, mode=None, position=None):
        if mode is not None and mode not in self.VALID_MODES:
            raise ValueError("Invalid tracking mode")
        if position is not None and position not in self.VALID_POSITIONS:
            raise ValueError("Invalid framing position")
        with self._lock:
            if mode is not None and mode != self._mode:
                self._mode = mode
                self._revision += 1
            if position is not None and position != self._position:
                self._position = position
                self._revision += 1
            return self._mode, self._position, self._revision


DEFAULT_TRACKING_SETTINGS = TrackingSettings()
POSITION_X_RATIOS = {"left": 0.25, "center": 0.50, "right": 0.75}


def corrected_ptz_target(target, frame_width, position):
    """Convert a desired off-center framing point into center-based PTZ error."""
    if target is None:
        return None
    desired_x = frame_width * POSITION_X_RATIOS[position]
    return (target[0] + frame_width / 2 - desired_x, target[1])


FAST_SPEED = 6
SLOW_SPEED = 2



# ============================================================
# PERSON TARGET POSITION
# ============================================================

# 사람 박스의 정중앙보다 조금 위쪽을 화면 중앙으로 가져옴
PERSON_TARGET_Y_RATIO = 0.38

# ============================================================
# AUTO ZOOM
# ============================================================

# 사람이 화면 세로 길이에서 차지해야 하는 목표 범위
# 우선 60~80%로 테스트
ZOOM_TARGET_MIN_RATIO = 0.68
ZOOM_TARGET_MAX_RATIO = 0.80

# 목표 중심값 0.78, 허용 범위 ±0.04
ZOOM_TARGET_RATIO = 0.74
ZOOM_TOLERANCE = 0.06

# 한 번에 조금씩 줌을 조절
ZOOM_ADJUST_TIME = 0.15

# 줌 명령 후 렌즈가 실제로 이동할 시간을 확보
ZOOM_SETTLE_TIME = 0.45

# GROUP 전환 시 줌아웃 시간
GROUP_ZOOM_TIME = 0.8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box[:4]

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def person_target(box):
    """PTZ가 실제로 따라갈 사람의 위치."""

    x1, y1, x2, y2 = box[:4]

    target_x = (x1 + x2) / 2

    # 사람 박스의 정중앙보다 조금 위
    target_y = y1 + (y2 - y1) * PERSON_TARGET_Y_RATIO

    return target_x, target_y

def get_person_height_ratio(person_box, frame_height):
    """
    사람 bounding box가 화면 높이에서 차지하는 비율.
    """

    if person_box is None or frame_height <= 0:
        return 0.0

    _, y1, _, y2 = person_box[:4]

    person_height = max(0, y2 - y1)

    return person_height / frame_height


def is_target_in_center_zone(
    target,
    frame_width,
    frame_height
):
    if target is None:
        return False

    tx, ty = target

    center_x = frame_width / 2
    center_y = frame_height / 2

    zone_half_width = frame_width * CENTER_ZONE_X
    zone_half_height = frame_height * CENTER_ZONE_Y

    return (
        center_x - zone_half_width
        <= tx
        <= center_x + zone_half_width

        and

        center_y - zone_half_height
        <= ty
        <= center_y + zone_half_height
    )


def is_target_in_stop_zone(
    target,
    frame_width,
    frame_height
):
    if target is None:
        return False

    tx, ty = target

    center_x = frame_width / 2
    center_y = frame_height / 2

    stop_half_width = frame_width * STOP_ZONE_X
    stop_half_height = frame_height * STOP_ZONE_Y

    return (
        center_x - stop_half_width
        <= tx
        <= center_x + stop_half_width

        and

        center_y - stop_half_height
        <= ty
        <= center_y + stop_half_height
    )

def is_target_in_stop_zone(
    target,
    frame_width,
    frame_height
):
    if target is None:
        return False

    tx, ty = target

    center_x = frame_width / 2
    center_y = frame_height / 2

    stop_half_width = frame_width * STOP_ZONE_X
    stop_half_height = frame_height * STOP_ZONE_Y

    return (
        center_x - stop_half_width
        <= tx
        <= center_x + stop_half_width

        and

        center_y - stop_half_height
        <= ty
        <= center_y + stop_half_height
    )


# ============================================================
# PTZ UNLOCK AREA
# ============================================================

def is_target_in_unlock_zone(
    target,
    frame_width,
    frame_height
):
    """
    PTZ가 한번 중앙에 맞은 뒤,
    사람이 이 영역 안에 있는 동안에는 카메라를 다시 움직이지 않는다.
    """

    if target is None:
        return False

    tx, ty = target

    center_x = frame_width / 2
    center_y = frame_height / 2

    half_width = (
        frame_width
        * PTZ_UNLOCK_ZONE_X
    )

    half_height = (
        frame_height
        * PTZ_UNLOCK_ZONE_Y
    )

    return (
        center_x - half_width
        <= tx
        <= center_x + half_width

        and

        center_y - half_height
        <= ty
        <= center_y + half_height
    )


# ============================================================
# PERSON TRACKING
# ============================================================

def detect_people(frame, person_model):
    """
    YOLO tracking으로 사람을 찾는다.

    return:
        {
            track_id: [x1, y1, x2, y2, confidence]
        }
    """

    people = {}

    results = person_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=PERSON_CONF,
        verbose=False,
    )[0]

    if results.boxes is None:
        return people

    for box in results.boxes:

        if box.id is None:
            continue

        track_id = int(box.id[0])

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        people[track_id] = [
            x1,
            y1,
            x2,
            y2,
            confidence
        ]

    return people


# ============================================================
# MICROPHONE DETECTION
# ============================================================

def detect_microphones_for_people(
    frame,
    people,
    mic_model
):
    """
    각 사람 ROI에서 마이크를 탐지한다.

    중요:
    마이크 위치 자체를 PTZ TARGET으로 사용하지 않는다.

    마이크가 어느 사람에게서 발견됐는지만 반환한다.
    """

    frame_height, frame_width = frame.shape[:2]

    detections = []

    for track_id, person_box in people.items():

        px1, py1, px2, py2 = map(
            int,
            person_box[:4]
        )

        rx1 = max(
            0,
            px1 - PADDING
        )

        ry1 = max(
            0,
            py1 - PADDING
        )

        rx2 = min(
            frame_width,
            px2 + PADDING
        )

        ry2 = min(
            frame_height,
            py2 + PADDING
        )

        roi = frame[
            ry1:ry2,
            rx1:rx2
        ]

        if roi.size == 0:
            continue

        mic_results = mic_model(
            roi,
            verbose=False
        )[0]

        # 한 사람 ROI 안에서 여러 개가 잡혀도
        # 가장 confidence가 높은 마이크 하나만 사용
        best_mic = None
        best_conf = 0.0

        for mic_box in mic_results.boxes:

            confidence = float(
                mic_box.conf[0]
            )

            if confidence < MIC_CONF:
                continue

            if confidence > best_conf:

                mx1, my1, mx2, my2 = map(
                    int,
                    mic_box.xyxy[0]
                )

                best_conf = confidence

                best_mic = [
                    rx1 + mx1,
                    ry1 + my1,
                    rx1 + mx2,
                    ry1 + my2,
                    confidence
                ]

        if best_mic is not None:

            detections.append(
                {
                    "track_id": track_id,
                    "mic": best_mic,
                    "confidence": best_conf,
                }
            )

    return detections


# ============================================================
# SPEAKER DECISION
# ============================================================

def choose_speaker(
    current_speaker_id,
    mic_hits,
    mic_scores
):
    """
    5초 동안 누적된 결과를 사용해서
    발화자를 결정한다.
    """

    if not mic_hits:
        return current_speaker_id

    candidates = []

    for track_id, hits in mic_hits.items():

        if hits < MIN_MIC_HITS:
            continue

        score = mic_scores.get(
            track_id,
            0.0
        )

        candidates.append(
            (
                track_id,
                hits,
                score
            )
        )

    if not candidates:
        print(
            "[SPEAKER] no reliable microphone candidate"
        )

        return current_speaker_id

    # 우선 검출 횟수,
    # 그다음 confidence 누적 점수
    candidates.sort(
        key=lambda item: (
            item[1],
            item[2]
        ),
        reverse=True
    )

    best_id, best_hits, best_score = candidates[0]

    # 아직 발화자가 없다면 바로 선정
    if current_speaker_id is None:

        print(
            f"[SPEAKER] selected ID {best_id} "
            f"(hits={best_hits}, score={best_score:.2f})"
        )

        return best_id

    # 현재 발화자가 계속 최고라면 유지
    if best_id == current_speaker_id:
        return current_speaker_id

    current_hits = mic_hits.get(
        current_speaker_id,
        0
    )

    current_score = mic_scores.get(
        current_speaker_id,
        0.0
    )

    # 새로운 사람이 충분히 강한 경우에만 변경
    best_total = (
        best_hits
        + best_score
    )

    current_total = (
        current_hits
        + current_score
    )

    if (
        best_total
        >= current_total
        + SPEAKER_SWITCH_MARGIN
    ):

        print(
            f"[SPEAKER] changed "
            f"{current_speaker_id} -> {best_id}"
        )

        return best_id

    print(
        f"[SPEAKER] keep ID {current_speaker_id}"
    )

    return current_speaker_id


# ============================================================
# DRAW
# ============================================================

def draw_results(
    frame,
    people,
    current_speaker_id,
    target,
    mic_detections,
    mic_search_active,
    search_remaining
):

    frame_height, frame_width = frame.shape[:2]

    center_x = frame_width // 2
    center_y = frame_height // 2

    # --------------------------------------------------------
    # 큰 CENTER 영역
    # --------------------------------------------------------

    center_half_width = int(
        frame_width * CENTER_ZONE_X
    )

    center_half_height = int(
        frame_height * CENTER_ZONE_Y
    )

    cv2.rectangle(
        frame,
        (
            center_x - center_half_width,
            center_y - center_half_height
        ),
        (
            center_x + center_half_width,
            center_y + center_half_height
        ),
        (255, 255, 0),
        2
    )

    # --------------------------------------------------------
    # 작은 STOP 영역
    # --------------------------------------------------------

    stop_half_width = int(
        frame_width * STOP_ZONE_X
    )

    stop_half_height = int(
        frame_height * STOP_ZONE_Y
    )

    cv2.rectangle(
        frame,
        (
            center_x - stop_half_width,
            center_y - stop_half_height
        ),
        (
            center_x + stop_half_width,
            center_y + stop_half_height
        ),
        (0, 0, 255),
        2
    )

    # --------------------------------------------------------
    # 사람 표시
    # --------------------------------------------------------

    for track_id, person_box in people.items():

        x1, y1, x2, y2 = map(
            int,
            person_box[:4]
        )

        thickness = 1

        if track_id == current_speaker_id:
            thickness = 3

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 100, 0),
            thickness
        )

        label = f"PERSON ID {track_id}"

        if track_id == current_speaker_id:
            label += " SPEAKER"

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 0),
            1
        )

    # --------------------------------------------------------
    # 마이크 표시
    # --------------------------------------------------------

    for detection in mic_detections:

        mic = detection["mic"]
        track_id = detection["track_id"]

        x1, y1, x2, y2 = map(
            int,
            mic[:4]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 80),
            2
        )

        cv2.putText(
            frame,
            f"MIC -> ID {track_id} {mic[4]:.2f}",
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 80),
            1
        )

    # --------------------------------------------------------
    # PTZ TARGET
    # --------------------------------------------------------

    if target is not None:

        tx, ty = map(
            int,
            target
        )

        cv2.drawMarker(
            frame,
            (tx, ty),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2
        )

    # --------------------------------------------------------
    # 상태 표시
    # --------------------------------------------------------

    if len(people) == 0:

        status = "NO PERSON"

    elif len(people) == 1:

        status = "SINGLE PERSON"

    elif mic_search_active:

        status = (
            "MIC SEARCH CONTINUING"
            if search_remaining <= 0.0
            else f"MIC SEARCH {search_remaining:.1f}s"
        )

    else:

        status = (
            f"MIC PAUSE "
            f"{search_remaining:.1f}s"
        )

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    speaker_text = (
        "NONE"
        if current_speaker_id is None
        else str(current_speaker_id)
    )

    cv2.putText(
        frame,
        f"PEOPLE: {len(people)} | SPEAKER: {speaker_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


# ============================================================
# MAIN
# ============================================================

def run(*, rtsp_url=None, ptz_controller=None, stop_event=None, show_debug=True, status_callback=None, tracking_settings=None):
    """Single tracking engine for CLI and FastAPI.

    A supplied PTZ controller is owned by the caller and is not closed here.
    """

    tracking_settings = tracking_settings or DEFAULT_TRACKING_SETTINGS

    # --------------------------------------------------------
    # 모델
    # --------------------------------------------------------

    person_model = YOLO(
        "yolov8n.pt"
    )

    mic_model = YOLO(
        MIC_MODEL_PATH
    )

    # --------------------------------------------------------
    # 카메라 / PTZ
    # --------------------------------------------------------

    camera = LatestFrameCamera(
        rtsp_url if rtsp_url is not None else RTSP_URL
    )

    owns_controller = ptz_controller is None
    controller = ptz_controller if ptz_controller is not None else ViscaPTZController()

    # --------------------------------------------------------
    # PTZ 상태
    # --------------------------------------------------------

    target_window = deque(
        maxlen=TARGET_WINDOW_SIZE
    )

    camera_moving = False

    # 카메라가 한 번 중앙에 맞춰진 후
    # 작은 YOLO 좌표 변화에는 반응하지 않도록 잠금
    ptz_locked = False

    # ============================================================
    # ZOOM / FRAMING STATE
    # ============================================================

    # 현재 줌이 맞춰져 있는 사람
    zoom_locked_person_id = None

    # 현재 GROUP 화면인지
    group_zoom_active = False

    # 새 발화자의 크기를 다시 확인해야 하는지
    zoom_needs_check = True
    zoom_centered_since = None

    # 마지막 줌 명령 시각. 연속 명령에 의한 오버슈트를 방지한다.
    last_zoom_adjust_time = 0.0

    # --------------------------------------------------------
    # 발화자 상태
    # --------------------------------------------------------

    current_speaker_id = None

    last_speaker_seen_time = 0.0

    # --------------------------------------------------------
    # 마이크 검색 상태
    # --------------------------------------------------------

    mic_search_active = False

    mic_search_start_time = None

    mic_pause_start_time = None

    last_mic_detect_time = 0.0

    mic_hits = defaultdict(int)

    mic_scores = defaultdict(float)

    current_mic_detections = []

    # 현재 마이크 탐색 구간에서
    # 신뢰할 수 있는 마이크 사용자가 몇 명인지 저장
    reliable_mic_person_ids = set()

    # 현재 GROUP 모드 여부
    group_mode = False

    previous_people_count = 0
    professor_id = None
    last_settings_revision = -1
    last_mode = None
    last_position = None

    try:

        if owns_controller:
            controller.connect()

        while stop_event is None or not stop_event.is_set():

            frame = camera.read()

            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()

            mode, position, revision = tracking_settings.snapshot()
            if revision != last_settings_revision:
                if camera_moving or mode == "fixed":
                    controller.pan_tilt_stop()
                    camera_moving = False
                if mode == "fixed":
                    controller.zoom_stop()
                target_window.clear()
                ptz_locked = False
                zoom_centered_since = None
                if mode != last_mode:
                    professor_id = None
                    current_speaker_id = None
                    previous_people_count = 0
                    group_mode = False
                    group_zoom_active = False
                    zoom_locked_person_id = None
                    zoom_needs_check = True
                    mic_search_active = False
                    mic_search_start_time = None
                    mic_pause_start_time = None
                    mic_hits.clear()
                    mic_scores.clear()
                    reliable_mic_person_ids.clear()
                    current_mic_detections = []
                last_settings_revision = revision
                last_mode = mode
                last_position = position
                print(f"[MODE] {mode.upper()} | POSITION {position.upper()}")

            # =================================================
            # 1. 사람 TRACKING
            # =================================================

            people = detect_people(
                frame,
                person_model
            )

            people_count = len(people)

            target = None

            # =================================================
            # 2. 사람이 없음
            # =================================================

            if mode == "fixed":
                current_speaker_id = None
                current_mic_detections = []
                mic_search_active = False
                target = None
            elif mode == "professor":
                current_mic_detections = []
                mic_search_active = False
                mic_search_start_time = None
                mic_pause_start_time = None
                mic_hits.clear()
                mic_scores.clear()
                group_mode = False
                group_zoom_active = False
                if professor_id is None and people:
                    # Deterministic first detection: highest-confidence person.
                    professor_id = max(people, key=lambda pid: people[pid][4])
                    print(f"[PROFESSOR] locked ID {professor_id}")
                    zoom_needs_check = True
                    zoom_locked_person_id = None
                    zoom_centered_since = None
                    ptz_locked = False
                current_speaker_id = professor_id
                if professor_id is not None and professor_id in people:
                    target = person_target(people[professor_id])
                    last_speaker_seen_time = now
                else:
                    # Never assign a different ID without explicit mode reset.
                    target = None
                    if camera_moving:
                        controller.pan_tilt_stop()
                        camera_moving = False
            else:
                if people_count == 0:

                    if camera_moving:
                        controller.pan_tilt_stop()
                        camera_moving = False

                    current_speaker_id = None

                    target_window.clear()

                    zoom_locked_person_id = None
                    zoom_needs_check = True

                    group_mode = False
                    group_zoom_active = False

                    mic_search_active = False
                    mic_search_start_time = None
                    mic_pause_start_time = None

                    mic_hits.clear()
                    mic_scores.clear()
                    reliable_mic_person_ids.clear()

                    current_mic_detections = []

                # =================================================
                # 3. 사람 한 명
                # =================================================

                elif people_count == 1:

                    # 유일한 사람
                    only_id = next(
                        iter(people)
                    )

                    # GROUP에서 다시 한 명이 된 경우 또는
                    # 다른 사람이 들어온 경우
                    if (
                        group_mode
                        or current_speaker_id != only_id
                    ):

                        print(
                            f"[ZOOM] SINGLE PERSON "
                            f"ID {only_id} -> CHECK SIZE"
                        )

                        group_mode = False
                        group_zoom_active = False

                        zoom_locked_person_id = None
                        zoom_needs_check = True
                        zoom_centered_since = None

                        ptz_locked = False

                        target_window.clear()

                    current_speaker_id = only_id

                    last_speaker_seen_time = now

                    # 마이크 탐색 필요 없음
                    mic_search_active = False
                    mic_search_start_time = None
                    mic_pause_start_time = None

                    mic_hits.clear()
                    mic_scores.clear()

                    current_mic_detections = []

                    # 사람 자체를 PTZ TARGET으로 사용
                    target = person_target(
                        people[only_id]
                    )

                # =================================================
                # 4. 사람 두 명 이상
                # =================================================

                else:

                    # ------------------------------------------------
                    # 방금 1명 이하 -> 2명 이상이 된 경우
                    # 즉시 마이크 탐색 시작
                    # ------------------------------------------------

                    if previous_people_count < 2:

                        mic_search_active = True

                        mic_search_start_time = now

                        mic_pause_start_time = None

                        last_mic_detect_time = 0.0

                        mic_hits.clear()
                        mic_scores.clear()

                        current_mic_detections = []

                        print(
                            "[MIC] SEARCH START"
                        )

                    # ------------------------------------------------
                    # 현재 발화자가 아직 화면에 있는지 확인
                    # ------------------------------------------------

                    if (
                        current_speaker_id is not None
                        and
                        current_speaker_id in people
                    ):

                        last_speaker_seen_time = now

                    elif (
                        current_speaker_id is not None
                        and
                        now - last_speaker_seen_time
                        > SPEAKER_LOST_TIMEOUT
                    ):

                        print(
                            f"[SPEAKER] ID "
                            f"{current_speaker_id} lost"
                        )

                        current_speaker_id = None

                        # 발화자를 잃어버렸다면
                        # 다음 검색을 기다리지 않고 바로 검색
                        mic_search_active = True

                        mic_search_start_time = now

                        mic_pause_start_time = None

                        last_mic_detect_time = 0.0

                        mic_hits.clear()
                        mic_scores.clear()

                    # =================================================
                    # MIC SEARCH 중
                    # =================================================

                    if mic_search_active:

                        elapsed = (
                            now
                            - mic_search_start_time
                        )

                        # ---------------------------------------------
                        # 일정 간격마다 마이크 탐지
                        # ---------------------------------------------

                        if (
                            now - last_mic_detect_time
                            >= MIC_DETECT_INTERVAL
                        ):

                            current_mic_detections = (
                                detect_microphones_for_people(
                                    frame,
                                    people,
                                    mic_model
                                )
                            )

                            last_mic_detect_time = now

                            # 사람별 결과 누적
                            for detection in current_mic_detections:

                                track_id = detection[
                                    "track_id"
                                ]

                                confidence = detection[
                                    "confidence"
                                ]

                                mic_hits[
                                    track_id
                                ] += 1

                                mic_scores[
                                    track_id
                                ] += confidence

                        # ---------------------------------------------
                        # 최소 3초 탐색 후 신뢰할 마이크가 있을 때만 종료
                        # 없으면 누적 결과를 유지하며 계속 탐색
                        # ---------------------------------------------

                        if elapsed >= MIC_SEARCH_DURATION and any(
                            hits >= MIN_MIC_HITS
                            for hits in mic_hits.values()
                        ):

                            # 최소 검출 횟수를 만족한 사람만
                            # 실제 마이크 사용자로 인정
                            reliable_mic_person_ids = {
                                track_id
                                for track_id, hits in mic_hits.items()
                                if hits >= MIN_MIC_HITS
                            }

                            reliable_mic_count = len(
                                reliable_mic_person_ids
                            )

                            print(
                                f"[MIC] reliable users = "
                                f"{reliable_mic_person_ids}"
                            )

                            # =====================================================
                            # 마이크 사용자 2명 이상 -> GROUP
                            # =====================================================

                            if reliable_mic_count >= 2:

                                if not group_mode:

                                    print(
                                        "[ZOOM] MULTIPLE MIC USERS "
                                        "-> GROUP"
                                    )

                                    controller.zoom_out(
                                        duration=GROUP_ZOOM_TIME
                                    )

                                group_mode = True

                                group_zoom_active = True

                                # 개인 Zoom Lock 해제
                                zoom_locked_person_id = None
                                zoom_needs_check = False

                                # GROUP에서는 특정 발화자를 추적하지 않음
                                current_speaker_id = None

                                target_window.clear()


                            # =====================================================
                            # 마이크 사용자 1명 -> 해당 사람
                            # =====================================================

                            elif reliable_mic_count == 1:

                                new_speaker_id = next(
                                    iter(reliable_mic_person_ids)
                                )

                                # GROUP에서 개인 화면으로 돌아오는 경우
                                if group_mode:

                                    print(
                                        f"[ZOOM] GROUP -> SPEAKER "
                                        f"ID {new_speaker_id}"
                                    )

                                    group_mode = False
                                    group_zoom_active = False

                                    zoom_locked_person_id = None
                                    zoom_needs_check = True
                                    zoom_centered_since = None

                                    ptz_locked = False

                                    target_window.clear()

                                # 발화자가 바뀐 경우
                                elif current_speaker_id != new_speaker_id:

                                    print(
                                        f"[SPEAKER] changed "
                                        f"{current_speaker_id} "
                                        f"-> {new_speaker_id}"
                                    )

                                    zoom_locked_person_id = None
                                    zoom_needs_check = True
                                    zoom_centered_since = None

                                    ptz_locked = False

                                    target_window.clear()

                                current_speaker_id = new_speaker_id


                            # =====================================================
                            # 신뢰할 마이크 사용자가 없음
                            # =====================================================

                            else:

                                print(
                                    "[MIC] no reliable microphone user"
                                )

                                # 기존 발화자가 화면에 있다면 그대로 유지
                                if (
                                    current_speaker_id is not None
                                    and current_speaker_id in people
                                ):

                                    print(
                                        f"[SPEAKER] keep "
                                        f"ID {current_speaker_id}"
                                    )

                                else:

                                    current_speaker_id = None

                            # 검색 종료
                            mic_search_active = False

                            mic_search_start_time = None

                            # 3초 휴식 시작
                            mic_pause_start_time = now

                            mic_hits.clear()
                            mic_scores.clear()

                            current_mic_detections = []

                            print(
                                "[MIC] SEARCH END "
                                "-> PAUSE"
                            )

                    # =================================================
                    # MIC PAUSE 중
                    # =================================================

                    else:

                        current_mic_detections = []

                        # 타이머가 없다면 지금부터 휴식
                        if mic_pause_start_time is None:
                            mic_pause_start_time = now

                        pause_elapsed = (
                            now
                            - mic_pause_start_time
                        )

                        # ---------------------------------------------
                        # 3초 휴식 종료
                        # ---------------------------------------------

                        if (
                            pause_elapsed
                            >= MIC_SEARCH_PAUSE
                        ):

                            mic_search_active = True

                            mic_search_start_time = now

                            mic_pause_start_time = None

                            last_mic_detect_time = 0.0

                            mic_hits.clear()
                            mic_scores.clear()

                            print(
                                "[MIC] PAUSE END "
                                "-> SEARCH"
                            )

                    # =================================================
                    # 현재 발화자를 PTZ TARGET으로 사용
                    # =================================================

                    if (
                        current_speaker_id is not None
                        and
                        current_speaker_id in people
                    ):

                        target = person_target(
                            people[
                                current_speaker_id
                            ]
                        )

            # =================================================
            if mode != "fixed":
                # 5. PTZ CONTROL
                # =================================================

                if target is not None:

                    target_window.append(
                        target
                    )

                    smoothed_target = np.mean(
                        target_window,
                        axis=0
                    )

                    frame_height, frame_width = (
                        frame.shape[:2]
                    )
                    control_target = corrected_ptz_target(smoothed_target, frame_width, position)

                    # =====================================================
                    # PTZ LOCK 상태 확인
                    # =====================================================

                    # 이미 PTZ가 중앙에 맞춰진 상태이고
                    # 사람이 UNLOCK 영역 안에 있다면
                    # 작은 움직임은 전부 무시
                    if (
                        ptz_locked
                        and is_target_in_unlock_zone(
                            control_target,
                            frame_width,
                            frame_height
                        )
                    ):

                        if camera_moving:
                            controller.pan_tilt_stop()
                            camera_moving = False

                        # 여기서는 아무 이동도 하지 않음


                    # -----------------------------------------------------
                    # LOCK 상태인데 사람이 UNLOCK 영역 밖으로 나감
                    # -----------------------------------------------------

                    elif ptz_locked:

                        print(
                            "[PTZ] TARGET MOVED -> UNLOCK"
                        )

                        ptz_locked = False

                        # 이 프레임에서는 잠금만 해제한다.
                        # 실제 이동은 다음 프레임부터 시작.


                    # =====================================================
                    # STOP 영역
                    # =====================================================

                    elif is_target_in_stop_zone(
                        control_target,
                        frame_width,
                        frame_height
                    ):

                        if camera_moving:

                            controller.pan_tilt_stop()

                            camera_moving = False

                        # 중앙에 충분히 들어왔으므로
                        # PTZ 위치 잠금
                        if not ptz_locked:

                            print(
                                "[PTZ] CENTERED -> LOCK"
                            )

                        ptz_locked = True


                    # =====================================================
                    # CENTER 영역
                    # 천천히 이동
                    # =====================================================

                    elif is_target_in_center_zone(
                        control_target,
                        frame_width,
                        frame_height
                    ):

                        moved = (
                            controller.move_to_target(
                                float(
                                    control_target[0]
                                ),
                                float(
                                    control_target[1]
                                ),
                                frame_width,
                                frame_height,
                                speed=SLOW_SPEED
                            )
                        )

                        if moved:
                            camera_moving = True


                    # =====================================================
                    # CENTER 밖
                    # 빠르게 이동
                    # =====================================================

                    else:

                        moved = (
                            controller.move_to_target(
                                float(
                                    control_target[0]
                                ),
                                float(
                                    control_target[1]
                                ),
                                frame_width,
                                frame_height,
                                speed=FAST_SPEED
                            )
                        )

                        if moved:
                            camera_moving = True

                    target = smoothed_target

                # 줌 조절은 대상이 화면 중앙에서 안정된 뒤에만 실행한다.
                if target is not None and is_target_in_stop_zone(
                    corrected_ptz_target(target, frame.shape[1], position), frame.shape[1], frame.shape[0]
                ) and not camera_moving:
                    if zoom_centered_since is None:
                        zoom_centered_since = now
                else:
                    zoom_centered_since = None

                # =================================================
                # AUTO ZOOM CONTROL
                # =================================================

                if (
                    zoom_centered_since is not None
                    and now - zoom_centered_since >= 0.35
                    and not group_mode
                    and zoom_needs_check
                    and current_speaker_id is not None
                    and current_speaker_id in people
                ):

                    frame_height = frame.shape[0]
                    person_box = people[current_speaker_id]

                    person_ratio = get_person_height_ratio(
                        person_box,
                        frame_height
                    )

                    print(
                        f"[ZOOM SINGLE] ID={current_speaker_id}, "
                        f"height_ratio={person_ratio:.3f}, "
                        f"target={ZOOM_TARGET_RATIO:.3f}, "
                        f"tolerance={ZOOM_TOLERANCE:.3f}, "
                        f"range={ZOOM_TARGET_MIN_RATIO:.3f}~{ZOOM_TARGET_MAX_RATIO:.3f}"
                    )

                    # 목표 범위에 들어오면 같은 발화자에 대해서는 줌을 잠근다.
                    if ZOOM_TARGET_MIN_RATIO <= person_ratio <= ZOOM_TARGET_MAX_RATIO:
                        zoom_locked_person_id = current_speaker_id
                        zoom_needs_check = False

                        print(
                            f"[AUTO ZOOM] LOCK ID={current_speaker_id}, "
                            f"ratio={person_ratio:.3f}"
                        )

                    # 직전 줌 명령 후 0.45초가 지나야 다음 줌 명령을 허용한다.
                    elif now - last_zoom_adjust_time >= ZOOM_SETTLE_TIME:

                        if person_ratio < ZOOM_TARGET_MIN_RATIO:
                            print(
                                f"[AUTO ZOOM IN] {person_ratio:.3f} "
                                f"< {ZOOM_TARGET_MIN_RATIO:.3f}"
                            )
                            controller.zoom_in(duration=ZOOM_ADJUST_TIME)
                            last_zoom_adjust_time = now

                        elif person_ratio > ZOOM_TARGET_MAX_RATIO:
                            print(
                                f"[AUTO ZOOM OUT] {person_ratio:.3f} "
                                f"> {ZOOM_TARGET_MAX_RATIO:.3f}"
                            )
                            controller.zoom_out(duration=ZOOM_ADJUST_TIME)
                            last_zoom_adjust_time = now

            # =================================================
            # 6. 화면 표시용 남은 시간
            # =================================================

            search_remaining = 0.0

            if people_count >= 2:

                if (
                    mic_search_active
                    and
                    mic_search_start_time is not None
                ):

                    search_remaining = max(
                        0.0,
                        MIC_SEARCH_DURATION
                        - (
                            now
                            - mic_search_start_time
                        )
                    )

                elif mic_pause_start_time is not None:

                    search_remaining = max(
                        0.0,
                        MIC_SEARCH_PAUSE
                        - (
                            now
                            - mic_pause_start_time
                        )
                    )

            # =================================================
            # 7. 화면 표시
            # =================================================

            draw_results(
                frame,
                people,
                current_speaker_id,
                target,
                current_mic_detections,
                mic_search_active,
                search_remaining
            )

            if status_callback is not None:
                status_callback({
                    "people_count": people_count,
                    "speaker_id": current_speaker_id,
                    "mode": mode.upper(),
                    "tracking_detail": "GROUP" if group_mode else ("SINGLE" if target is not None else "IDLE"),
                    "position": position,
                    "professor_id": professor_id if mode == "professor" else None,
                    "mic_state": "SEARCH" if mic_search_active else "PAUSE" if mode == "multi" and people_count >= 2 else "OFF",
                    "zoom_target_range": "68–80%",
                })
            if show_debug:
                cv2.imshow("YOLO Speaker PTZ Control", frame)

            # 현재 사람 수 저장
            previous_people_count = people_count

            if show_debug and (cv2.waitKey(1) & 0xFF in (27, ord("q"))):
                break

    except (
        OSError,
        RuntimeError
    ) as exc:

        print(
            "[ERROR] PTZ control failed:",
            exc
        )

    finally:

        try:
            controller.pan_tilt_stop()
        except Exception:
            pass

        try:
            if owns_controller:
                controller.close()
        except Exception:
            pass

        camera.stop()

        if show_debug:
            cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    run()
