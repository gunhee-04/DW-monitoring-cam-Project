from flask import Flask, Response, render_template_string, render_template
import cv2
import requests
import time
from ultralytics import YOLO
import base64

app = Flask(__name__)

# =========================
# 설정
# =========================
PC1_VIDEO_URL = "http://192.168.0.9:5000/video"   # PC1 주소로 변경
PC3_API_URL   = "http://192.168.0.9:8090/api/detection"  # PC3 주소로 변경 / 관제 서버로 넘기는것(restapi로 받을때!) 

CAMERA_ID = "CAM-01"
SEND_INTERVAL = 2.0   # 초마다 PC3로 전송

# YOLO 모델 로드(YOLOv8 nano 모델 로드)
model = YOLO("yolov8n.pt")

# MJPEG 스트림 연결(PC1 영상 연결)
cap = cv2.VideoCapture(PC1_VIDEO_URL)

last_sent_time = 0


def send_detection_data(objects, frame):        # 역할: AI 분석 결과 > PC3 서버 전송

    image_base64 = encode_image(frame)          # 이미지 Base64 변환

    payload = {                     # 전송 데이터 생성
        "cameraId": CAMERA_ID,
        "timestamp": int(time.time() * 1000),
        "image": image_base64,
        "objects": objects
    }

    try:
        response = requests.post(PC3_API_URL, json=payload, timeout=2)      # HTTP POST 전송
        print(f"[PC2] PC3 전송 성공: {response.status_code}")
    except Exception as e:
        print(f"[PC2] PC3 전송 실패: {e}")


def gen_frames():                       # AI 분석
    global last_sent_time

    while True:
        success, frame = cap.read()     # PC1 영상 읽기(PC1 영상 수신)
        if not success:
            print("[PC2] PC1 영상 읽기 실패")
            time.sleep(1)
            continue

        # YOLO 객체 탐지 + 추적
        results = model.track(frame, persist=True, verbose=False)

        # 객체 정보 생성(각 객체 정보를 저장)
        objects = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:       # 객체 반복
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1      # 클래스 추출
                label = model.names.get(cls_id, "unknown")

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())     # 좌표 추출

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                confidence = None
                if box.conf is not None:
                    confidence = float(box.conf[0].item())

                objects.append({                # 객체 정보 저장-관제 서버로 보낼 데이터
                    "id": track_id,
                    "type": label,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

        # 일정 시간마다 관제서버(PC3)로 전송
        current_time = time.time()
        
        # 박스가 그려진 화면 생성
        annotated_frame = results[0].plot() if results else frame

        if current_time - last_sent_time >= SEND_INTERVAL:
            send_detection_data(objects, annotated_frame)
            last_sent_time = current_time

        ret, buffer = cv2.imencode(".jpg", annotated_frame)         # 영상 스트림을 위한 JPEG 압축
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG 스트리밍
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

'''
이미지 Base64 변환: 이미지 Base64 변환
이유: HTTP JSON 전송 가능
'''
def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text


@app.route("/")
def index():
    return render_template("index.html")
    '''
    return render_template_string("""
    <html>
    <head>
        <title>PC2 AI Streaming Server</title>
    </head>
    <body>
        <h2>PC2 AI Streaming Server</h2>
        <p>Source: PC1 video stream</p>
        
        <img src="/video" width="900">
    </body>
    </html>
    """)
    '''


@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)