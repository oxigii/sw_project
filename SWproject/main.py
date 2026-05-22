from ultralytics import YOLO
import cv2
import argparse
import os

person_model = YOLO("yolov8n.pt")
mic_model = YOLO("runs/detect/train/weights/best.pt")

PADDING = 20
PERSON_CONF = 0.5
MIC_CONF = 0.5


def process_frame(frame):
    display = frame.copy()
    h, w = frame.shape[:2]
    mic_detected_in_frame = False

    person_results = person_model(frame, verbose=False)[0]

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

            abs_mx1 = rx1 + mx1
            abs_my1 = ry1 + my1
            abs_mx2 = rx1 + mx2
            abs_my2 = ry1 + my2

            mic_cx = (abs_mx1 + abs_mx2) // 2
            mic_cy = (abs_my1 + abs_my2) // 2

            cv2.rectangle(
                display,
                (abs_mx1, abs_my1),
                (abs_mx2, abs_my2),
                (0, 255, 80),
                2
            )

            cv2.circle(display, (mic_cx, mic_cy), 5, (0, 255, 80), -1)

            cv2.putText(
                display,
                f"MIC {float(mbox.conf):.2f}",
                (abs_mx1, abs_my1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 80),
                1
            )

            print(
                f"  마이크 감지: conf={float(mbox.conf):.2f}, "
                f"중심=({mic_cx},{mic_cy})"
            )

    return display, mic_detected_in_frame


def run_image(source, save):
    frame = cv2.imread(source)

    if frame is None:
        print(f"[ERROR] 이미지를 불러올 수 없습니다: {source}")
        return

    print(f"[이미지 모드] {source}")

    result, mic_detected = process_frame(frame)

    total_frames = 1
    mic_detected_frames = 1 if mic_detected else 0

    print(f"[결과] 총 프레임 수: {total_frames}")
    print(f"[결과] 마이크 인식 프레임 수: {mic_detected_frames}")
    print(f"[결과] 마이크 인식 비율: {mic_detected_frames / total_frames * 100:.2f}%")

    if save:
        os.makedirs("videos", exist_ok=True)

        base = os.path.basename(source)
        name, ext = os.path.splitext(base)
        save_path = f"videos/{name}_result{ext}"

        cv2.imwrite(save_path, result)
        print(f"[저장] {save_path}")

    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(source, save):
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[ERROR] 소스를 열 수 없습니다: {source}")
        return

    mode = "카메라" if str(source).isdigit() else "영상"
    print(f"[{mode} 모드] {source}  |  Q 키 종료")

    out = None
    save_path = None

    if save:
        os.makedirs("videos", exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30

        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not str(source).isdigit():
            base = os.path.basename(source)
            name, ext = os.path.splitext(base)
            save_path = f"videos/{name}_result{ext}"
        else:
            save_path = "videos/camera_result.mp4"

        out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (vw, vh)
        )

        print(f"[저장] {save_path} 로 기록 중...")

    total_frames = 0
    mic_detected_frames = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        result, mic_detected = process_frame(frame)

        if mic_detected:
            mic_detected_frames += 1

        cv2.putText(
            result,
            f"MIC frames: {mic_detected_frames}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if out:
            out.write(result)

        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"[결과] 총 프레임 수: {total_frames}")
    print(f"[결과] 마이크 인식 프레임 수: {mic_detected_frames}")

    if total_frames > 0:
        ratio = mic_detected_frames / total_frames * 100
        print(f"[결과] 마이크 인식 비율: {ratio:.2f}%")

    cap.release()

    if out:
        out.release()
        print(f"[저장 완료] {save_path}")

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="2단계 YOLO 마이크 감지")

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="입력 소스: 이미지 경로 / 영상 경로 / 카메라 인덱스(0,1,...)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="결과 저장"
    )

    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.5,
        help="1단계 사람 감지 신뢰도 임계값"
    )

    parser.add_argument(
        "--mic-conf",
        type=float,
        default=0.4,
        help="2단계 마이크 감지 신뢰도 임계값"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    PERSON_CONF = args.person_conf
    MIC_CONF = args.mic_conf

    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    if is_image:
        run_image(source, args.save)
    else:
        run_video(source, args.save)