"""Streamlit dashboard for the speaker-aware PTZ camera project."""

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from PTZ_control.ptz_worker import PTZTrackingWorker, WorkerConfig


PROJECT_ROOT = Path(__file__).resolve().parent
PTZ_DIR = PROJECT_ROOT / "PTZ_control"
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_RTSP_URL = os.getenv("PTZ_RTSP_URL", "")
DEFAULT_CAMERA_IP = os.getenv("PTZ_CAMERA_IP", "")


st.set_page_config(
    page_title="발화자 인식 PTZ 제어",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e6eaf0;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .camera-placeholder {
        min-height: 460px;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        background: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #64748b;
        text-align: center;
        font-size: 1.05rem;
    }
    .mode-card {
        border-radius: 12px;
        padding: 14px 16px;
        background: #eef6ff;
        border: 1px solid #cfe4ff;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "worker" not in st.session_state:
    st.session_state.worker = None
if "auto_tracking" not in st.session_state:
    st.session_state.auto_tracking = True


def worker_snapshot():
    worker = st.session_state.worker
    if worker is None:
        return None, {
            "running": False,
            "status": "대기",
            "camera_connected": False,
            "source_mode": "webcam",
            "mode": "IDLE",
            "mic_count": 0,
            "person_count": 0,
            "fps": 0.0,
            "tracking_enabled": st.session_state.auto_tracking,
            "error": None,
            "logs": [],
        }
    return worker.snapshot()


def send_manual_command(command):
    worker = st.session_state.worker
    if worker is None:
        st.warning("먼저 시스템을 시작해 주세요.")
        return

    success, message = worker.manual_control(command)
    if success:
        st.toast(message, icon="✅")
    else:
        st.warning(message)


st.title("발화자 인식 기반 PTZ 카메라")
st.caption("마이크 1개는 단일 추적, 2개 이상은 그룹 화면으로 자동 전환합니다.")

with st.sidebar:
    st.header("연결 및 검출 설정")
    source_label = st.radio(
        "카메라 모드",
        options=["노트북 캠 테스트", "PTZ 카메라"],
        horizontal=True,
    )
    source_mode = "webcam" if source_label == "노트북 캠 테스트" else "ptz"

    rtsp_url = ""
    camera_ip = ""
    visca_port = 5678
    webcam_index = 0
    digital_zoom = 1.6

    if source_mode == "webcam":
        webcam_index = st.number_input(
            "노트북 카메라 번호",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="내장 카메라는 일반적으로 0번입니다.",
        )
        digital_zoom = st.slider(
            "디지털 추적 확대 배율",
            min_value=1.1,
            max_value=2.5,
            value=1.6,
            step=0.1,
        )
        st.info("노트북 캠 모드에서는 VISCA 연결 없이 디지털 화면 추적을 사용합니다.")
    else:
        rtsp_url = st.text_input(
            "RTSP 주소",
            value=DEFAULT_RTSP_URL,
        )
        camera_ip = st.text_input("카메라 IP", value=DEFAULT_CAMERA_IP)
        visca_port = st.number_input(
            "VISCA 포트",
            min_value=1,
            max_value=65535,
            value=5678,
            step=1,
        )

    mic_model_path = st.text_input(
        "마이크 모델 경로",
        value=str(PTZ_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"),
    )

    with st.expander("세부 설정"):
        person_conf = st.slider(
            "사람 신뢰도",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
        mic_conf = st.slider(
            "마이크 신뢰도",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
        )
        confirm_frames = st.slider(
            "모드 전환 확인 프레임",
            min_value=3,
            max_value=30,
            value=10,
        )
        window_size = st.slider(
            "좌표 안정화 Window",
            min_value=1,
            max_value=15,
            value=5,
        )
        dead_zone_x = 120
        dead_zone_y = 90
        ptz_speed = 3
        if source_mode == "ptz":
            dead_zone_x = st.slider(
                "좌우 Dead Zone",
                min_value=20,
                max_value=250,
                value=120,
                step=10,
            )
            dead_zone_y = st.slider(
                "상하 Dead Zone",
                min_value=20,
                max_value=200,
                value=90,
                step=10,
            )
            ptz_speed = st.slider(
                "PTZ 속도",
                min_value=1,
                max_value=10,
                value=3,
            )

    st.info("설정값은 시스템을 다시 시작할 때 적용됩니다.")


_, current_state = worker_snapshot()
is_running = bool(current_state["running"])

start_col, stop_col, clear_col = st.columns([1, 1, 1.3])

with start_col:
    if st.button(
        "▶ 시스템 시작",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    ):
        config = WorkerConfig(
            source_mode=source_mode,
            rtsp_url=rtsp_url.strip(),
            mic_model_path=mic_model_path.strip(),
            camera_ip=camera_ip.strip(),
            webcam_index=int(webcam_index),
            digital_zoom=float(digital_zoom),
            visca_port=int(visca_port),
            person_conf=float(person_conf),
            mic_conf=float(mic_conf),
            mode_confirm_frames=int(confirm_frames),
            target_window_size=int(window_size),
            dead_zone_x=int(dead_zone_x),
            dead_zone_y=int(dead_zone_y),
            ptz_speed=int(ptz_speed),
        )
        worker = PTZTrackingWorker(config)
        worker.set_tracking_enabled(st.session_state.auto_tracking)
        st.session_state.worker = worker
        worker.start()
        st.rerun()

with stop_col:
    if st.button(
        "■ 시스템 중지",
        use_container_width=True,
        disabled=not is_running,
    ):
        st.session_state.worker.stop()
        st.rerun()

with clear_col:
    if st.button("↺ 화면 초기화", use_container_width=True):
        worker = st.session_state.worker
        if worker is not None:
            worker.stop()
            worker.clear_screen()
        st.session_state.worker = None
        st.session_state.auto_tracking = True
        st.rerun()


@st.fragment(run_every=0.25)
def live_dashboard():
    frame, state = worker_snapshot()
    mode_labels = {
        "IDLE": "대기",
        "SINGLE": "단일 발화자 추적",
        "GROUP": "그룹 화면",
    }

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("시스템 상태", state["status"])
    metric_2.metric("현재 모드", mode_labels.get(state["mode"], state["mode"]))
    metric_3.metric("검출", f"사람 {state['person_count']} / 마이크 {state['mic_count']}")
    metric_4.metric("처리 속도", f"{state['fps']:.1f} FPS")

    video_col, status_col = st.columns([3, 1])

    with video_col:
        st.subheader("실시간 분석 화면")
        if frame is None:
            st.markdown(
                """
                <div class="camera-placeholder">
                    시스템 시작을 누르면<br>
                    카메라 영상과 YOLO 검출 결과가 표시됩니다.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(
                frame,
                channels="BGR",
                # use_container_width=True,
                caption="사람: 파란 박스 · 마이크: 초록 박스 · 추적 중심: 빨간 십자",
            )

    with status_col:
        st.subheader("동작 상태")
        connection = "연결됨" if state["camera_connected"] else "연결 안 됨"
        tracking = "ON" if state["tracking_enabled"] else "OFF"
        source_name = "노트북 캠" if state["source_mode"] == "webcam" else "PTZ 카메라"
        st.markdown(
            f"""
            <div class="mode-card">
                <b>카메라</b>: {connection}<br>
                <b>입력 모드</b>: {source_name}<br>
                <b>자동 추적</b>: {tracking}<br>
                <b>제어 모드</b>: {mode_labels.get(state["mode"], state["mode"])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if state["error"]:
            st.error(state["error"])
        elif state["mode"] == "GROUP":
            st.info("마이크가 여러 개 감지되어 전체 화면을 유지합니다.")
        elif state["mode"] == "SINGLE":
            st.success("한 명의 마이크를 중심으로 추적 중입니다.")
        elif state["running"]:
            st.warning("마이크가 감지되지 않아 현재 위치를 유지합니다.")

        with st.expander("실행 로그", expanded=False):
            if state["logs"]:
                st.code("\n".join(state["logs"]), language=None)
            else:
                st.caption("아직 실행 기록이 없습니다.")


live_dashboard()

st.divider()
control_title, tracking_col = st.columns([2, 1])
active_source_mode = current_state["source_mode"] if is_running else source_mode
with control_title:
    st.subheader("카메라 제어")
    if active_source_mode == "webcam":
        st.caption("노트북 캠에서는 자동 추적 시 디지털 크롭·확대를 적용합니다.")
    else:
        st.caption("자동 추적을 끄면 방향·줌·초기 위치를 수동으로 조작할 수 있습니다.")

with tracking_col:
    st.toggle(
        "자동 추적",
        key="auto_tracking",
        disabled=not is_running,
    )

worker = st.session_state.worker
if worker is not None and is_running:
    _, latest_state = worker.snapshot()
    if latest_state["tracking_enabled"] != st.session_state.auto_tracking:
        worker.set_tracking_enabled(st.session_state.auto_tracking)

is_webcam_mode = active_source_mode == "webcam"
manual_disabled = (
    not is_running
    or st.session_state.auto_tracking
    or is_webcam_mode
)
left_controls, center_controls, right_controls = st.columns([1, 1, 1])

with left_controls:
    st.markdown("**방향 이동**")
    up_left, up, up_right = st.columns(3)
    with up:
        if st.button("▲", use_container_width=True, disabled=manual_disabled):
            send_manual_command("up")

    mid_left, mid_center, mid_right = st.columns(3)
    with mid_left:
        if st.button("◀", use_container_width=True, disabled=manual_disabled):
            send_manual_command("left")
    with mid_center:
        st.button("●", use_container_width=True, disabled=True)
    with mid_right:
        if st.button("▶", use_container_width=True, disabled=manual_disabled):
            send_manual_command("right")

    down_left, down, down_right = st.columns(3)
    with down:
        if st.button("▼", use_container_width=True, disabled=manual_disabled):
            send_manual_command("down")

with center_controls:
    st.markdown("**화면 확대/축소**")
    if st.button("＋ 줌인", use_container_width=True, disabled=manual_disabled):
        send_manual_command("zoom_in")
    if st.button("－ 줌아웃", use_container_width=True, disabled=manual_disabled):
        send_manual_command("zoom_out")

with right_controls:
    st.markdown("**초기 위치**")
    if st.button(
        "⌂ 카메라 초기 위치",
        use_container_width=True,
        disabled=manual_disabled,
    ):
        send_manual_command("reset_view")
    if is_webcam_mode:
        st.caption("노트북 캠은 물리적인 방향·줌 제어를 지원하지 않습니다.")
    else:
        st.caption("홈 위치로 이동한 뒤 광각 화면으로 전환합니다.")
