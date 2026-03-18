from flask import Flask,Response, render_template_string,render_template
import cv2

app = Flask(__name__)

#카메라 연결
CAMERA_INDEX=0
camera = cv2.VideoCapture(CAMERA_INDEX)

@app.route("/")
def index():
    ''' 주석
    return render_template_string("""
    <html>
          <head>
               <title>PC1 Camera Sender</title>             
          </head>
          <body>
                <h2>PC1 Camera Sender</h2>
                <img src="/video" width="800">
          </body>
                                  
    </html>
    """)
    '''

    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    while True:
        #success=boolean / frame=image
        success, frame = camera.read()
        if not success:
            break

        ret, buffer=cv2.imencode('.jpg',frame)  #openCV이미지 > JPEG 압축 후 buffer에 jpeg이미지 저장
        if not ret:
            continue

        frame_bytes=buffer.tobytes()
        #yield의미: generator 함수로 데이터를 계속 스트림으로 전송 함(한 번에 보내는 것이 아니라 프레임마다 계속 전송)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
