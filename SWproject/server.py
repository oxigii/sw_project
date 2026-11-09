"""FastAPI server for the standalone HTML PTZ control dashboard.



Run with \`\`uvicorn server:app --host 127.0.0.1 --port 8000\`\` and open

http://127.0.0.1:8000. The dashboard and API share one origin, so no CORS

configuration is required.

"""



from collections.abc import Iterator

import os
from pathlib import Path

from threading import Lock, Thread, Event

import time

from dotenv import load_dotenv
import numpy as np

from ultralytics import YOLO



from PTZ_control import yolo_rtsp_mic_detection as detection



import cv2

from fastapi import FastAPI, HTTPException

from fastapi.responses import FileResponse, StreamingResponse

from pydantic import BaseModel, Field



from PTZ_control.visca_ptz_control import ViscaPTZController





PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_RTSP_URL = os.getenv("PTZ_RTSP_URL", "")
DEFAULT_CAMERA_IP = os.getenv("PTZ_CAMERA_IP", "")

app = FastAPI(title="PTZ Camera Control API")



controller_lock = Lock()

controller: ViscaPTZController | None = None

auto_tracking = False

tracking_thread: Thread | None = None

tracking_stop_event = Event()



current_rtsp_url: str | None = None





class ConnectionRequest(BaseModel):

  camera_ip: str = Field(default="", max_length=255)

  visca_port: int = Field(default=5678, ge=1, le=65535)

  ptz_speed: int = Field(default=3, ge=1, le=24)

  rtsp_url: str = ""



class CommandRequest(BaseModel):

  command: str





class ModeRequest(BaseModel):
    mode: str | None = None
    position: str | None = None


class TrackingRequest(BaseModel):

  enabled: bool





def close_controller() -> None:

  """Close the active VISCA connection while holding \`\`controller_lock\`\`."""

  global controller

  if controller is not None:

    controller.close()

    controller = None





def require_controller() -> ViscaPTZController:

  if controller is None:

    raise HTTPException(status_code=409, detail="PTZ 카메라가 연결되지 않았습니다.")

  return controller





def mjpeg_frames(rtsp_url: str) -> Iterator[bytes]:

  """Encode the latest RTSP frames as a browser-compatible MJPEG stream."""

  capture = cv2.VideoCapture(rtsp_url)

  capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  try:

    if not capture.isOpened():

      return



    while True:

      success, frame = capture.read()

      if not success:

        break

      encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

      if not encoded:

        continue

      yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"

  finally:

    capture.release()





@app.get("/")

def dashboard() -> FileResponse:

  return FileResponse(PROJECT_ROOT / "ptz_dashboard.html")





@app.get("/api/status")

def status() -> dict:

  return {

    "connected": controller is not None,

    "auto_tracking": auto_tracking,
    "tracking": tracking_status,
    "settings": dict(zip(("mode", "position", "revision"), detection.DEFAULT_TRACKING_SETTINGS.snapshot())),

  }







@app.post("/api/mode")
def set_mode(data: ModeRequest) -> dict:
    """Apply mode immediately; fixed mode homes the camera and allows manual PTZ."""
    global auto_tracking
    old_mode, _, _ = detection.DEFAULT_TRACKING_SETTINGS.snapshot()
    # Validate before touching the active camera.
    if data.mode is not None and data.mode not in detection.TrackingSettings.VALID_MODES:
        raise HTTPException(status_code=422, detail="잘못된 추적 모드입니다.")
    if data.position is not None and data.position not in detection.TrackingSettings.VALID_POSITIONS:
        raise HTTPException(status_code=422, detail="잘못된 사람 위치입니다.")
    requested_mode = data.mode or old_mode
    if requested_mode == "fixed" and data.mode == "fixed":
        # Stop and join worker before sending home command. No concurrent auto PTZ.
        tracking_stop_event.set()
        if tracking_thread is not None and tracking_thread.is_alive():
            tracking_thread.join(timeout=8.0)
            if tracking_thread.is_alive():
                raise HTTPException(status_code=409, detail="자동 추적 종료 중입니다. 잠시 후 다시 눌러주세요.")
        auto_tracking = False
        with controller_lock:
            if controller is not None:
                try:
                    controller.pan_tilt_stop()
                    controller.zoom_stop()
                    controller.reset_view()
                except (OSError, RuntimeError) as exc:
                    raise HTTPException(status_code=502, detail=f"홈 위치 이동 실패: {exc}") from exc
    mode, position, revision = detection.DEFAULT_TRACKING_SETTINGS.update(
        mode=data.mode, position=data.position
    )
    # Choosing a tracking mode while connected activates it without another click.
    if mode != "fixed" and controller is not None and current_rtsp_url:
        set_tracking(TrackingRequest(enabled=True))
    return {"ok": True, "mode": mode, "position": position,
            "revision": revision, "auto_tracking": auto_tracking}

@app.post("/api/connect")

def connect_camera(data: ConnectionRequest) -> dict:
  if tracking_thread is not None and tracking_thread.is_alive():
    raise HTTPException(status_code=409, detail="자동 추적을 중지한 뒤 카메라를 재연결하세요.")

  global controller, current_rtsp_url

  rtsp_url = data.rtsp_url.strip() or DEFAULT_RTSP_URL
  camera_ip = data.camera_ip.strip() or DEFAULT_CAMERA_IP

  if not rtsp_url.startswith(("rtsp://", "rtsps://")):
    raise HTTPException(status_code=400, detail="RTSP 주소를 입력하거나 .env의 PTZ_RTSP_URL을 설정하세요.")
  if not camera_ip:
    raise HTTPException(status_code=400, detail="카메라 IP를 입력하거나 .env의 PTZ_CAMERA_IP를 설정하세요.")



  with controller_lock:

    try:

      close_controller()



      new_controller = ViscaPTZController(

        camera_ip=camera_ip,

        port=data.visca_port,

        speed=data.ptz_speed,

      )



      new_controller.connect()



      controller = new_controller

      current_rtsp_url = rtsp_url



    except OSError as exc:

      raise HTTPException(

        status_code=502,

        detail=f"카메라 연결 실패: {exc}"

      ) from exc



  return {

    "ok": True,

    "message": "PTZ 카메라가 연결되었습니다."

  }



@app.post("/api/disconnect")

def disconnect_camera() -> dict:
  if tracking_thread is not None and tracking_thread.is_alive():
    raise HTTPException(status_code=409, detail="자동 추적을 중지한 뒤 연결을 해제하세요.")

  with controller_lock:

    close_controller()

  return {"ok": True, "message": "PTZ 카메라 연결을 종료했습니다."}



tracking_status = {"people_count": 0, "speaker_id": None, "mode": "IDLE", "mic_state": "OFF", "zoom_target_range": "68–80%"}


def tracking_loop():
    """Run the same YOLO/PTZ engine used by the standalone entrypoint."""
    global auto_tracking
    active = controller
    url = current_rtsp_url
    if active is None or not url:
        auto_tracking = False
        return
    try:
        detection.run(
            rtsp_url=url,
            ptz_controller=active,
            stop_event=tracking_stop_event,
            show_debug=True,
            status_callback=lambda data: tracking_status.update(data),
            tracking_settings=detection.DEFAULT_TRACKING_SETTINGS,
        )
    except Exception as exc:
        print("[TRACKING ERROR]", repr(exc))
    finally:
        auto_tracking = False
        tracking_stop_event.set()
        print("[TRACKING] worker stopped")


@app.post("/api/tracking")
def set_tracking(data: TrackingRequest) -> dict:
    global auto_tracking, tracking_thread
    if data.enabled:
        if detection.DEFAULT_TRACKING_SETTINGS.snapshot()[0] == "fixed":
            raise HTTPException(status_code=409, detail="고정 모드에서는 자동 추적을 실행하지 않습니다.")
        if controller is None or not current_rtsp_url:
            raise HTTPException(status_code=409, detail="먼저 PTZ 카메라를 연결해주세요.")
        if tracking_thread is None or not tracking_thread.is_alive():
            tracking_stop_event.clear()
            auto_tracking = True
            tracking_thread = Thread(target=tracking_loop, daemon=True, name="fastapi-ptz-tracking")
            tracking_thread.start()
        else:
            auto_tracking = True
    else:
        auto_tracking = False
        tracking_stop_event.set()
        if tracking_thread is not None and tracking_thread.is_alive():
            tracking_thread.join(timeout=3.0)
        # Worker owns the PTZ stop command while it is running.
        if tracking_thread is None or not tracking_thread.is_alive():
            with controller_lock:
                if controller is not None:
                    controller.pan_tilt_stop()
    return {"ok": True, "auto_tracking": auto_tracking}


@app.post("/api/ptz")

def control_ptz(data: CommandRequest) -> dict:

  if auto_tracking and detection.DEFAULT_TRACKING_SETTINGS.snapshot()[0] != "fixed":

    raise HTTPException(status_code=409, detail="고정 모드 또는 자동 추적 OFF에서 수동 제어할 수 있습니다.")



  with controller_lock:

    active_controller = require_controller()

    try:

      if data.command in {"left", "right", "up", "down"}:

        active_controller.move_manual(data.command)

      elif data.command == "zoom_in":

        active_controller.zoom_in(duration=0.8)



      elif data.command == "zoom_out":

        active_controller.zoom_out(duration=0.8)

      elif data.command == "reset_view":

        active_controller.reset_view()

      else:

        raise HTTPException(status_code=400, detail="지원하지 않는 PTZ 명령입니다.")

    except (OSError, RuntimeError, ValueError) as exc:

      raise HTTPException(status_code=502, detail=f"PTZ 명령 실패: {exc}") from exc



  return {"ok": True, "command": data.command, "timestamp": time.time()}





@app.get("/api/video/stream")

def video_stream() -> StreamingResponse:
  rtsp_url = current_rtsp_url
  if not rtsp_url:
    raise HTTPException(status_code=409, detail="먼저 카메라를 연결하세요.")

  if not rtsp_url.startswith(("rtsp://", "rtsps://")):

    raise HTTPException(status_code=400, detail="RTSP 주소만 사용할 수 있습니다.")

  return StreamingResponse(

    mjpeg_frames(rtsp_url),

    media_type="multipart/x-mixed-replace; boundary=frame",

  )
