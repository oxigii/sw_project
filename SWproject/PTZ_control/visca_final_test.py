import os
import socket
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
CAMERA_IP = os.getenv("PTZ_CAMERA_IP", "")

# 가장 많이 쓰는 VISCA over IP 포트
VISCA_PORT = 5678

def send(sock, hex_cmd):
    cmd = bytes.fromhex(hex_cmd)
    sock.sendall(cmd)

try:
    print("VISCA TCP 연결 시도...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    sock.connect((CAMERA_IP, VISCA_PORT))

    print("연결 성공")

    # 왼쪽 이동
    print("왼쪽 이동")
    send(sock, "81 01 06 01 08 08 01 03 FF")
    time.sleep(1)

    # 정지
    print("정지")
    send(sock, "81 01 06 01 00 00 03 03 FF")
    time.sleep(1)

    # 오른쪽 이동
    print("오른쪽 이동")
    send(sock, "81 01 06 01 08 08 02 03 FF")
    time.sleep(1)

    # 정지
    print("정지")
    send(sock, "81 01 06 01 00 00 03 03 FF")

    print("테스트 완료")

except Exception as e:
    print("VISCA 실패:", e)

finally:
    try:
        sock.close()
    except:
        pass