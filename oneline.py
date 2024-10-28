from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from vidgear.gears import CamGear
from tracker import *

app = Flask(__name__)

model = YOLO('yolov8s.pt')
stream = CamGear(source='https://www.youtube.com/watch?v=FsL_KQz4gpw', stream_mode=True, logging=True).start()

# Global variables for car counting
car_count = 0
tracked_cars = {}

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    global car_count, tracked_cars
    # Lines for counting
    line_1_y = 300
    line_2_y = 300
    tolerance = 6

    # Load COCO class names
    with open("coco.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    tracker = Tracker()

    while True:
        frame = stream.read()
        if frame is None:
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame)
        a = results[0].boxes.data
        print(a)

        bbox_list = []

        for row in a:
            print(row)
            bbox_x1 = int(row[0])
            bbox_y1 = int(row[1])
            bbox_x2 = int(row[2])
            bbox_y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'car' in c:
                bbox_list.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2])

        bbox_id = tracker.update(bbox_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            center_x = int(x3 + x4) // 2
            center_y = int(y3 + y4) // 2

            # Check if car crosses the defined lines
            if line_1_y - tolerance < center_y < line_1_y + tolerance:
                tracked_cars[id] = 'line 1'
            elif line_2_y - tolerance < center_y < line_2_y + tolerance:
                tracked_cars[id] = 'line 2'

            # Draw car position and ID
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(id), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Update car count
        car_count = len(tracked_cars)

        # Draw counting lines and car count
        cv2.line(frame, (550, line_1_y), (700, line_1_y), (255, 255, 255), 1)
        cv2.line(frame, (330, line_2_y), (470, line_2_y), (255, 255, 255), 1)
        cv2.putText(frame, f"car_count: {car_count}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
