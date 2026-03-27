from flask import Flask, Response, render_template
import cv2
import requests
import time
from ultralytics import YOLO
import base64
import os
import numpy as np
import threading

app = Flask(__name__)

# =========================
# 서버 설정
# =========================
SPRING_CONFIG_URL = "http://localhost:8081/api/cameras/{cameraCode}/config"
PC3_API_URL = "http://192.168.0.144:8081/api/events"

SOURCES = {
    "webcam": "http://192.168.0.144:5000/video",   # 기존 웹캠 송출 Flask(MJPEG)
    "drone1": "videos/crowd1.mp4",
    "drone2": "videos/danger1.mp4"
}

CAMERA_MAP = {
    "webcam": "CAM-02",
    "drone1": "CAM-02",
    "drone2": "CAM-02"
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
# 동기화/공유 상태
# =========================
state_lock = threading.Lock()
roi_lock = threading.Lock()
frame_lock = threading.Lock()
detect_lock = threading.Lock()

source_version = 0
latest_jpeg = None
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
# Spring 설정 조회
# =========================
def load_camera_config(camera_code):
    try:
        url = SPRING_CONFIG_URL.format(cameraCode=camera_code)
        r = requests.get(url, timeout=3)

        print(f"[설정 조회 요청] {url}")
        print(f"[설정 조회 응답코드] {r.status_code}")
        print(f"[설정 조회 응답본문] {r.text}")

        r.raise_for_status()
        data = r.json()

        intrusion_time = int(data["intrusionSeconds"])
        roi = data["roi"]

        roi_x1 = int(roi["x1"])
        roi_y1 = int(roi["y1"])
        roi_x2 = int(roi["x2"])
        roi_y2 = int(roi["y2"])

        return intrusion_time, roi_x1, roi_y1, roi_x2, roi_y2

    except Exception as e:
        print(f"[설정 로드 실패] {e}")
        return 3, 100, 100, 500, 400


# =========================
# 유틸
# =========================
def apply_camera_config(camera_id):
    global INTRUSION_TIME, ROI_POLYGON, roi_ready

    intrusion_time, x1, y1, x2, y2 = load_camera_config(camera_id)

    with roi_lock:
        INTRUSION_TIME = intrusion_time
        ROI_POLYGON = np.array([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ], np.int32)
        roi_ready = True

    print(
        f"[설정 적용 완료] camera={camera_id} / "
        f"intrusion={INTRUSION_TIME}s / roi=({x1},{y1})~({x2},{y2})"
    )


def reset_detection_state():
    with detect_lock:
        intruded_ids.clear()
        current_intruded_ids.clear()
        intrusion_enter_time.clear()
        snapshot_saved_ids.clear()


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

    with detect_lock:
        intrusion_now = len(current_intruded_ids)

    event_type = None
    event_level = None
    message = None
    stay_duration_sec = 0

    if intrusion_now > 0:
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
        "event_type": event_type,
        "event_level": event_level,
        "detectedCount": people_count,
        "stayDurationSec": stay_duration_sec,
        "message": message,
        "eventTime": int(time.time()),
        "image": image_base64
    }

    print("[전송 payload]", payload)

    try:
        r = requests.post(PC3_API_URL, json=payload, timeout=2)
        print(f"[전송 성공] {r.status_code}")
    except Exception as e:
        print(f"[전송 실패] {e}")


def ensure_default_roi(frame):
    global ROI_POLYGON, roi_ready

    with roi_lock:
        if roi_ready and ROI_POLYGON is not None:
            return ROI_POLYGON.copy(), INTRUSION_TIME

    h, w = frame.shape[:2]
    roi_w = int(w * 0.5)
    roi_h = int(h * 0.5)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    default_roi = np.array([
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2)
    ], np.int32)

    with roi_lock:
        if ROI_POLYGON is None or not roi_ready:
            ROI_POLYGON = default_roi
            roi_ready = True
            print("[ROI] 기본 중앙 영역 설정 완료")
        return ROI_POLYGON.copy(), INTRUSION_TIME


# =========================
# 입력 소스 생성/종료
# =========================
def create_reader(source_name):
    source_value = SOURCES[source_name]

    # webcam: MJPEG HTTP 스트림은 requests로 직접 읽음
    if source_name == "webcam" and isinstance(source_value, str) and source_value.startswith("http"):
        response = requests.get(source_value, stream=True, timeout=5)
        response.raise_for_status()
        return {
            "mode": "mjpeg",
            "response": response,
            "iterator": response.iter_content(chunk_size=1024),
            "buffer": b""
        }

    # mp4 등은 cv2.VideoCapture 사용
    cap = cv2.VideoCapture(source_value)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source_name}")
    return {
        "mode": "cv2",
        "cap": cap
    }


def close_reader(reader):
    if reader is None:
        return

    try:
        if reader["mode"] == "cv2":
            reader["cap"].release()
        elif reader["mode"] == "mjpeg":
            reader["response"].close()
    except Exception:
        pass


def read_frame_from_reader(reader, source_name):
    if reader["mode"] == "cv2":
        cap = reader["cap"]
        success, frame = cap.read()

        # mp4 끝나면 처음부터 반복
        if (not success or frame is None) and source_name != "webcam":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()

        if not success or frame is None:
            return None

        return frame

    # MJPEG 수동 파싱 (boundary 문제 회피)
    while True:
        chunk = next(reader["iterator"], None)
        if chunk is None:
            raise RuntimeError("mjpeg stream ended")

        reader["buffer"] += chunk

        start = reader["buffer"].find(b"\xff\xd8")
        end = reader["buffer"].find(b"\xff\xd9")

        if start != -1 and end != -1 and end > start:
            jpg = reader["buffer"][start:end + 2]
            reader["buffer"] = reader["buffer"][end + 2:]

            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame


# =========================
# 시작 시 최초 설정 적용
# =========================
apply_camera_config(CAMERA_ID)


# =========================
# 소스 전환 API
# =========================
@app.route("/switch/<source>")
def switch_source(source):
    global CURRENT_SOURCE, CAMERA_ID, source_version, frame_count, last_sent_time

    if source not in SOURCES:
        return f"invalid source: {source}", 400

    new_camera_id = CAMERA_MAP[source]

    reset_detection_state()
    apply_camera_config(new_camera_id)

    with state_lock:
        CURRENT_SOURCE = source
        CAMERA_ID = new_camera_id
        frame_count = 0
        last_sent_time = 0
        source_version += 1

    print(f"[소스 변경] {source} / CAMERA_ID={CAMERA_ID}")
    return f"switched to {source}"


# =========================
# 설정 수동 새로고침 API
# =========================
@app.route("/reload-config")
def reload_config():
    reset_detection_state()
    apply_camera_config(CAMERA_ID)
    return {
        "status": "ok",
        "cameraId": CAMERA_ID,
        "intrusionTime": INTRUSION_TIME
    }


# =========================
# 캡처 + YOLO 백그라운드 루프
# =========================
def capture_loop():
    global latest_jpeg, last_sent_time, frame_count

    local_version = -1
    reader = None
    reader_source = None
    last_results = None

    while True:
        with state_lock:
            current_version = source_version
            current_source = CURRENT_SOURCE

        if local_version != current_version or reader is None or reader_source != current_source:
            close_reader(reader)
            reader = None
            last_results = None

            try:
                reader = create_reader(current_source)
                reader_source = current_source
                local_version = current_version
                frame_count = 0
                print(f"[스트림 전환] {current_source}")
            except Exception as e:
                print(f"[소스 열기 실패] {current_source} / {e}")
                time.sleep(1)
                continue

        try:
            frame = read_frame_from_reader(reader, current_source)
        except Exception as e:
            print(f"[프레임 읽기 실패] {current_source} / {e}")
            close_reader(reader)
            reader = None
            time.sleep(0.3)
            continue

        if frame is None:
            time.sleep(0.02)
            continue

        local_roi, local_intrusion_time = ensure_default_roi(frame)

        frame_count += 1

        frame_small = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        scale_x = frame.shape[1] / RESIZE_W
        scale_y = frame.shape[0] / RESIZE_H

        if frame_count % FRAME_SKIP == 0:
            try:
                last_results = model.track(
                    frame_small,
                    persist=True,
                    verbose=False,
                    classes=[PERSON_CLASS_ID],
                    conf=0.5,
                    imgsz=YOLO_IMGSZ
                )
            except Exception as e:
                print(f"[YOLO 예외] {e}")

        results = last_results

        person_count = 0
        annotated_frame = frame.copy()

        with detect_lock:
            current_intruded_ids.clear()

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
                    inside = cv2.pointPolygonTest(local_roi, (cx, cy), False) >= 0

                    intrusion = False
                    if track_id is not None:
                        if inside:
                            if track_id not in intrusion_enter_time:
                                intrusion_enter_time[track_id] = time.time()

                            stay = time.time() - intrusion_enter_time[track_id]

                            if stay >= local_intrusion_time:
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

            intrusion_now = len(current_intruded_ids)
            intrusion_total = len(intruded_ids)

        cv2.polylines(annotated_frame, [local_roi], True, (0, 0, 255), 2)

        cv2.putText(
            annotated_frame,
            f"Source: {current_source}",
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
            f"Intrusion(Now): {intrusion_now}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.putText(
            annotated_frame,
            f"Intrusion(Total): {intrusion_total}",
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

        with frame_lock:
            latest_jpeg = buffer.tobytes()


capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()


# =========================
# 스트리밍
# =========================
def gen_frames():
    while True:
        with frame_lock:
            jpg = latest_jpeg

        if jpg is None:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpg +
            b"\r\n"
        )

        time.sleep(0.01)


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