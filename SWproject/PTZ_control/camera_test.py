"""
camera_test.py

YOLO + PTZ 추적 프로그램을 OpenCV 창에서 직접 실행하는 진입점.

기존 yolo_rtsp_mic_detection.py의 run()을 그대로 사용한다.
"""

import sys
from pathlib import Path


# ============================================================
# 현재 파일이 있는 PTZ_control 폴더를 Python import 경로에 추가
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


# ============================================================
# 기존 YOLO + PTZ 프로그램 불러오기
# ============================================================

try:
    from yolo_rtsp_mic_detection import run

except ImportError as e:
    print()
    print("==============================================")
    print("[ERROR] yolo_rtsp_mic_detection.py import 실패")
    print("==============================================")
    print()
    print(e)
    print()
    print("camera_test.py와 yolo_rtsp_mic_detection.py가")
    print("같은 PTZ_control 폴더에 있는지 확인하세요.")
    print()

    raise


# ============================================================
# 프로그램 시작
# ============================================================

def main():

    print()
    print("==============================================")
    print(" YOLO + PTZ OpenCV Test")
    print("==============================================")
    print()
    print("[INFO] OpenCV 직접 실행 모드")
    print("[INFO] 기존 YOLO 모델 사용")
    print("[INFO] 기존 PTZ 제어 사용")
    print("[INFO] 기존 마이크 검출 사용")
    print()
    print("[INFO] 종료 방법")
    print("       Q   : 종료")
    print("       ESC : 종료")
    print()
    print("==============================================")
    print()

    try:

        run()

    except KeyboardInterrupt:

        print()
        print("[INFO] KeyboardInterrupt")
        print("[INFO] 프로그램을 종료합니다.")

    except Exception as e:

        print()
        print("==============================================")
        print("[ERROR] 프로그램 실행 중 오류 발생")
        print("==============================================")
        print()

        print(type(e).__name__)
        print(e)

        print()

        raise

    finally:

        print()
        print("[INFO] camera_test 종료")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
