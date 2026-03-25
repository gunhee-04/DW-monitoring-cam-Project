from flask import Flask, Response, render_template
import cv2
import requests
import time
from ultralytics import YOLO
import base64
import os
import numpy as np

app = Flask(__name__)

# =========================
# 서버 설정
# =========================
PC3_API_URL = "http://192.168.0.144:8081/api/detection"

SOURCES = {
    "webcam": "http://192.168.0.144:5000/video",   # 기존 웹캠 송출 Flask
    "drone1": "videos/crowd1.mp4",
    "drone2": "videos/danger1.mp4"
}

CAMERA_MAP = {
    "webcam": "CAM-01",
    "drone1": "CAM-02",
    "drone2": "CAM-03"
}

CURRENT_SOURCE = "webcam"
CAMERA_ID = CAMERA_MAP[CURRENT_SOURCE]

SEND_INTERVAL = 2.0
INTRUSION_TIME = 3.0
CROWD_THRESHOLD = 5

FRAME_SKIP = 3
RESIZE_W = 640
RESIZE_H = 360
YOLO_IMGSZ = 416

# =========================
# YOLO
# =========================
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

# =========================
# 영상 연결
# =========================
cap = cv2.VideoCapture(SOURCES[CURRENT_SOURCE])
last_sent_time = 0
frame_count = 0

# =========================
# ROI
# =========================
ROI_POLYGON = None
roi_ready = False

# =========================
# 침입 관리
# =========================
intruded_ids = set()
current_intruded_ids = set()
intrusion_enter_time = {}
snapshot_saved_ids = set()

os.makedirs("intrusion_snapshots", exist_ok=True)

# =========================
# 유틸
# =========================
def reset_detection_state():
    global ROI_POLYGON, roi_ready
    intruded_ids.clear()
    current_intruded_ids.clear()
    intrusion_enter_time.clear()
    snapshot_saved_ids.clear()
    ROI_POLYGON = None
    roi_ready = False


def encode_image(frame):
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def save_snapshot(frame, track_id):
    filename = f"intrusion_snapshots/intrusion_{track_id}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[침입 스냅샷 저장] {filename}")


def send_detection_data(frame, people_count):
    global CAMERA_ID

    event_type = None
    event_level = None
    message = None
    stay_duration_sec = 0

    if len(current_intruded_ids) > 0:
        event_type = "INTRUSION"
        event_level = "HIGH"
        message = "위험구역 침입 감지"
        stay_duration_sec = int(INTRUSION_TIME)
    elif people_count >= CROWD_THRESHOLD:
        event_type = "CROWD"
        event_level = "MEDIUM"
        message = "밀집 발생"
    else:
        return

    image_base64 = encode_image(frame)

    payload = {
        "cameraId": CAMERA_ID,
        "eventType": event_type,
        "eventLevel": event_level,
        "detectedCount": people_count,
        "stayDurationSec": stay_duration_sec,
        "message": message,
        "eventTime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_base64
    }

    print("[전송 payload]", payload)

    try:
        r = requests.post(PC3_API_URL, json=payload, timeout=2)
        print(f"[전송 성공] {r.status_code}")
    except Exception as e:
        print(f"[전송 실패] {e}")


# =========================
# 소스 전환 API
# =========================
@app.route("/switch/<source>")
def switch_source(source):
    global cap, CURRENT_SOURCE, CAMERA_ID

    if source not in SOURCES:
        return f"invalid source: {source}", 400

    CURRENT_SOURCE = source
    CAMERA_ID = CAMERA_MAP[source]

    cap.release()
    cap = cv2.VideoCapture(SOURCES[source])
    reset_detection_state()

    print(f"[소스 변경] {source} / CAMERA_ID={CAMERA_ID}")
    return f"switched to {source}"


# =========================
# 스트리밍
# =========================
def gen_frames():
    global last_sent_time, frame_count, ROI_POLYGON, roi_ready, cap

    while True:
        success, frame = cap.read()

        # mp4 끝나면 처음부터 반복
        if not success or frame is None:
            if CURRENT_SOURCE != "webcam":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                time.sleep(0.05)
                continue

        if not roi_ready:
            h, w = frame.shape[:2]
            roi_w = int(w * 0.5)
            roi_h = int(h * 0.5)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h

            ROI_POLYGON = np.array([
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ], np.int32)

            roi_ready = True
            print("[ROI] 화면 중앙 영역 설정 완료")

        frame_count += 1

        frame_small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        scale_x = frame.shape[1] / RESIZE_W
        scale_y = frame.shape[0] / RESIZE_H

        results = None
        if frame_count % FRAME_SKIP == 0:
            results = model.track(
                frame_small,
                persist=True,
                verbose=False,
                classes=[PERSON_CLASS_ID],
                conf=0.5,
                imgsz=YOLO_IMGSZ
            )

        person_count = 0
        current_intruded_ids.clear()
        annotated_frame = frame.copy()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                track_id = int(box.id[0]) if box.id is not None else None
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                inside = cv2.pointPolygonTest(ROI_POLYGON, (cx, cy), False) >= 0

                intrusion = False
                if track_id is not None:
                    if inside:
                        if track_id not in intrusion_enter_time:
                            intrusion_enter_time[track_id] = time.time()

                        stay = time.time() - intrusion_enter_time[track_id]

                        if stay >= INTRUSION_TIME:
                            intrusion = True
                            current_intruded_ids.add(track_id)
                            intruded_ids.add(track_id)

                            if track_id not in snapshot_saved_ids:
                                save_snapshot(frame, track_id)
                                snapshot_saved_ids.add(track_id)
                    else:
                        intrusion_enter_time.pop(track_id, None)

                color = (0, 0, 255) if intrusion else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"ID:{track_id} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.polylines(annotated_frame, [ROI_POLYGON], True, (0, 0, 255), 2)

        cv2.putText(
            annotated_frame,
            f"Source: {CURRENT_SOURCE}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"People: {person_count}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Intrusion(Now): {len(current_intruded_ids)}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Intrusion(Total): {len(intruded_ids)}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        now = time.time()
        if now - last_sent_time >= SEND_INTERVAL:
            send_detection_data(annotated_frame, person_count)
            last_sent_time = now

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)