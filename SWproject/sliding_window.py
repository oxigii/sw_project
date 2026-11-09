import cv2
import argparse
import os
import sys
import numpy as np
from collections import deque
from ultralytics import YOLO
from frame_capture import save_detected_frame

# ── 모델 로드 ──────────────────────────────────────────────
person_model = YOLO("yolo11n.pt")
mic_model = YOLO("runs/detect/train/weights/best.pt")

PADDING = 5
PERSON_CONF = 0.5
MIC_CONF = 0.5

# ── 슬라이딩 윈도우 설정 ───────────────────────────────────
WINDOW_SIZE = 5
window_mic_dict = {}


def make_result_path(source):
    """
    원본 파일과 같은 위치에 파일명_result 형태로 저장
    예: main.mp4 -> main_result.mp4
    예: test.jpg -> test_result.jpg
    """
    if str(source).isdigit():
        return "camera_result.mp4"

    folder = os.path.dirname(source)
    base = os.path.basename(source)
    name, ext = os.path.splitext(base)

    save_name = f"{name}_result{ext}"
    return os.path.join(folder, save_name)


def update_sliding_window(detected_boxes):
    global window_mic_dict
    new_window_dict = {}

    for box in detected_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        best_id = None
        min_dist = 100

        for mid, box_queue in window_mic_dict.items():
            if not box_queue:
                continue

            last_box = box_queue[-1]
            lcx = (last_box[0] + last_box[2]) / 2
            lcy = (last_box[1] + last_box[3]) / 2

            dist = np.sqrt((cx - lcx) ** 2 + (cy - lcy) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_id = mid

        if best_id is not None:
            new_queue = window_mic_dict[best_id]
            new_queue.append(box)
            new_window_dict[best_id] = new_queue
        else:
            new_id = len(new_window_dict) + len(window_mic_dict)
            new_queue = deque(maxlen=WINDOW_SIZE)
            new_queue.append(box)
            new_window_dict[new_id] = new_queue

    window_mic_dict = new_window_dict

    smoothed_boxes = []

    for mid, box_queue in window_mic_dict.items():
        mean_box = np.mean(box_queue, axis=0).tolist()
        smoothed_boxes.append(mean_box)

    return smoothed_boxes


def process_frame(frame):
    display = frame.copy()
    h, w = frame.shape[:2]

    mic_detected_in_frame = False

    person_results = person_model(frame, verbose=False)[0]
    current_frame_mics = []

    for pbox in person_results.boxes:
        if int(pbox.cls) != 0 or float(pbox.conf) < PERSON_CONF:
            continue

        px1, py1, px2, py2 = map(int, pbox.xyxy[0])

        rx1 = max(0, px1 - PADDING)
        ry1 = max(0, py1 - PADDING)
        rx2 = min(w, px2 + PADDING)
        ry2 = min(h, py2 + PADDING)

        roi = frame[ry1:ry2, rx1:rx2]
        mic_results = mic_model(roi, verbose=False)[0]

        cv2.rectangle(display, (px1, py1), (px2, py2), (255, 100, 0), 1)

        for mbox in mic_results.boxes:
            if float(mbox.conf) < MIC_CONF:
                continue

            mic_detected_in_frame = True

            mx1, my1, mx2, my2 = map(int, mbox.xyxy[0])

            ax1 = rx1 + mx1
            ay1 = ry1 + my1
            ax2 = rx1 + mx2
            ay2 = ry1 + my2

            current_frame_mics.append([ax1, ay1, ax2, ay2])

    smoothed_mics = update_sliding_window(current_frame_mics)

    for sbox in smoothed_mics:
        sx1, sy1, sx2, sy2 = map(int, sbox)

        cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 80), 2)

        cv2.putText(
            display,
            "MIC (Window)",
            (sx1, sy1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 80),
            1
        )

    return display, mic_detected_in_frame


def run(source, save):
    global window_mic_dict

    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        window_mic_dict = {}

        frame = cv2.imread(source)

        if frame is None:
            print(f"[ERROR] 이미지 로드 실패: {source}")
            return

        result, mic_detected = process_frame(frame)

        total_frames = 1
        mic_detected_frames = 1 if mic_detected else 0

        print(f"[결과] 총 프레임 수: {total_frames}")
        print(f"[결과] 마이크 인식 프레임 수: {mic_detected_frames}")
        print(f"[결과] 마이크 인식 비율: {mic_detected_frames / total_frames * 100:.2f}%")

        if save:
            save_path = make_result_path(source)
            cv2.imwrite(save_path, result)
            print(f"[저장 완료] {save_path}")

        print("[DONE] 이미지 처리 완료")

    else:
        window_mic_dict = {}

        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)

        if not cap.isOpened():
            print(f"[ERROR] 소스 로드 실패: {source}")
            return

        total_expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        save_path = None

        if save:
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 30

            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            save_path = make_result_path(source)

            out = cv2.VideoWriter(
                save_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (vw, vh)
            )

            print(f"[저장] {save_path} 로 기록 중...")

        frame_count = 0
        mic_detected_frames = 0

        print("[INFO] 분석 시작")

        if total_expected_frames > 0:
            print(f"[INFO] 영상 총 프레임 수: {total_expected_frames}")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                result, mic_detected = process_frame(frame)

                if mic_detected:
                    mic_detected_frames += 1
                    save_detected_frame(result)

                cv2.putText(
                    result,
                    f"MIC frames: {mic_detected_frames}/{frame_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                if out:
                    out.write(result)

                if total_expected_frames > 0:
                    percent = (frame_count / total_expected_frames) * 100
                    sys.stdout.write(
                        f"\r >>> 진행 상황: {percent:.1f}% "
                        f"({frame_count}/{total_expected_frames}) | "
                        f"MIC 인식 프레임: {mic_detected_frames}"
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(
                        f"\r >>> 처리 프레임: {frame_count} | "
                        f"MIC 인식 프레임: {mic_detected_frames}"
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n[STOP] 중단됨")

        cap.release()

        if out:
            out.release()

        print()
        print(f"[결과] 총 처리 프레임 수: {frame_count}")
        print(f"[결과] 마이크 인식 프레임 수: {mic_detected_frames}")

        if frame_count > 0:
            ratio = mic_detected_frames / frame_count * 100
            print(f"[결과] 마이크 인식 비율: {ratio:.2f}%")

        if save:
            print(f"[저장 완료] {save_path}")

        print("[DONE] 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이딩 윈도우 적용 2단계 YOLO 마이크 감지")

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="입력 소스: 이미지 경로 / 영상 경로 / 카메라 인덱스"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="결과 저장"
    )

    args = parser.parse_args()

    run(args.source, args.save)

# python mic_detect_sliding_window.py --source "main.mp4" --save    # 실행 명령어

# 웹캠(0번)으로 실시간 테스트하며 결과를 저장하고 싶을 때: python mic_detect_sliding_window.py --source 0 --save
# 저장하지 않고 화면 확인만 하고 싶을 때 (이미지나 영상 분석): python mic_detect_sliding_window.py --source your_file.jpg
