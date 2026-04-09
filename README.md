# DW-monitoring-cam-Project

## 프로젝트 소개
DW-monitoring-cam-Project는 카메라 영상을 수집하고, YOLOv8 기반 객체 탐지를 수행한 뒤, 탐지 결과를 관제 서버로 전달하는 AI 영상 처리 프로젝트입니다.

웹캠 영상을 송출하는 Camera Sender, 실시간 영상 분석 및 이벤트 전송을 담당하는 Streaming Server 구조로 분리하여 구성하였으며, Flask 기반 MJPEG 스트리밍 방식으로 실시간 영상을 제공합니다.

---

## 주요 기능
- 웹캠 영상을 실시간으로 송출
- YOLOv8 기반 객체 탐지 및 추적
- 탐지 결과를 이미지와 메타데이터 형태로 관제 서버에 전송
- Flask 기반 실시간 영상 스트리밍 제공
- 향후 관제 시스템 및 관리자 서버와 연동 가능한 구조로 설계

---

## 프로젝트 구조
```bash
DW-monitoring-cam-Project
├── camera_sender/        # 웹캠 영상 송출 서버
├── streaming_server/     # YOLO 기반 영상 분석 및 이벤트 전송 서버
├── flask-server/         # Flask 실험/확장용 디렉토리
├── spring-server/        # Spring 연동용 디렉토리
└── docs/                 # 문서 및 참고 자료
