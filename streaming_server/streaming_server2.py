from flask import Flask, Response, render_template
import cv2
import requests
import time
from ultralytics import YOLO
import base64
import os

app = Flask(__name__)

# =========================
# 서버 설정
# =========================
PC1_VIDEO_URL = "http://192.168.0.28:5000/video"
SPRING_CONFIG_URL = "http://localhost:8081/api/cameras/{cameraCode}/config"
PC3_API_URL = "http://192.168.0.9:8090/api/detection"

CAMERA_ID = "CAM-02"
SEND_INTERVAL = 2.0

# ===== 속도 최적화 옵션 =====
FRAME_SKIP = 3
RESIZE_W = 640
RESIZE_H = 360
YOLO_IMGSZ = 416

# =========================
# Spring 설정 조회
# =========================
def load_camera_config(camera_code):
    try:
        url = SPRING_CONFIG_URL.format(cameraCode=camera_code)
        r = requests.get(url, timeout=3)

        print("status:", r.status_code)
        print("text:", r.text)

        r.raise_for_status()

        data = r.json()

        print("==== JSON 응답 ====")
        print(data)

        intrusion_time = int(data["intrusionSeconds"])

        roi = data["roi"]

        roi_x1 = int(roi["x1"])
        roi_y1 = int(roi["y1"])
        roi_x2 = int(roi["x2"])
        roi_y2 = int(roi["y2"])

        return intrusion_time, roi_x1, roi_y1, roi_x2, roi_y2

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[설정 로드 실패] {e}")

        return 3, 100, 100, 500, 400


# ✅ 이 줄이 꼭 있어야 함
INTRUSION_TIME, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = load_camera_config(CAMERA_ID)
print(f"[설정 적용 완료] intrusion={INTRUSION_TIME}s / roi=({ROI_X1},{ROI_Y1}) ~ ({ROI_X2},{ROI_Y2})")

# =========================
# YOLO
# =========================
model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0

# =========================
# 영상 연결
# =========================
cap = cv2.VideoCapture(PC1_VIDEO_URL)
last_sent_time = 0
frame_count = 0

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
def encode_image(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def save_snapshot(frame, track_id):
    filename = f"intrusion_snapshots/intrusion_{track_id}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"[침입 스냅샷 저장] {filename}")


def send_detection_data(objects, frame, people_count):
    payload = {
        "cameraId": CAMERA_ID,
        "timestamp": int(time.time() * 1000),
        "peopleCount": people_count,
        "intrusionNow": len(current_intruded_ids),
        "intrusionTotal": len(intruded_ids),
        "image": encode_image(frame),
        "objects": objects
    }

    try:
        r = requests.post(PC3_API_URL, json=payload, timeout=2)
        print(f"[전송 성공] {r.status_code}")
    except Exception as e:
        print(f"[전송 실패] {e}")


def is_inside_roi(cx, cy):
    return ROI_X1 <= cx <= ROI_X2 and ROI_Y1 <= cy <= ROI_Y2


# =========================
# 스트리밍
# =========================
def gen_frames():
    global last_sent_time, frame_count

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            time.sleep(0.05)
            continue

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

        objects = []
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
                inside = is_inside_roi(cx, cy)

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

                objects.append({
                    "id": track_id,
                    "type": "person",
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "intrusion": intrusion
                })

        # ROI 사각형 표시
        cv2.rectangle(
            annotated_frame,
            (ROI_X1, ROI_Y1),
            (ROI_X2, ROI_Y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"People: {person_count}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Intrusion(Now): {len(current_intruded_ids)}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Intrusion(Total): {len(intruded_ids)}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        now = time.time()
        if now - last_sent_time >= SEND_INTERVAL:
            send_detection_data(objects, annotated_frame, person_count)
            last_sent_time = now

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )


# =========================
# Flask 라우팅
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)