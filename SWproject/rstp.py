import os
from pathlib import Path

import cv2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
rtsp_url = os.getenv("PTZ_RTSP_URL", "")
if not rtsp_url:
    raise RuntimeError(".env에 PTZ_RTSP_URL을 설정하세요.")

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("RTSP connection failed")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("frame read fail")
        break

    cv2.imshow("PTZ Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()