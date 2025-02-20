import cv2
import torch
import time
import asyncio
import websockets
import json
import sys
from ultralytics import YOLO
from flask import Flask, Response
from threading import Thread

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = YOLO("best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    cap.release()
    sys.exit()

app = Flask(__name__)
clients = set()
prev_time = time.time()

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

async def send_predictions(websocket):
    global prev_time
    clients.add(websocket)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            results = model.predict(frame, device=device)
            prediction = results[0]
            top1_idx = prediction.probs.top1
            top1_conf = float(prediction.probs.top1conf)
            top1_label = prediction.names[top1_idx]

            data = json.dumps({
                "gesture": top1_label,
                "confidence": round(top1_conf, 2)
            })

            await websocket.send(data)
            await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    finally:
        clients.remove(websocket)

async def websocket_server():
    async with websockets.serve(send_predictions, "0.0.0.0", 5001):
        await asyncio.Future()

def start_flask_app():
    app.run(host="0.0.0.0", port=5002)

if __name__ == "__main__":
    print("Starting Flask server for video feed on http://localhost:5002/video_feed")
    print("WebSocket Server running on ws://localhost:5001")

    flask_thread = Thread(target=start_flask_app)
    flask_thread.start()

    try:
        asyncio.run(websocket_server())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        cap.release()
        cv2.destroyAllWindows()
