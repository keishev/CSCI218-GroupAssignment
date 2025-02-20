import cv2
import torch
import time
import asyncio
import websockets
import json
import sys
from ultralytics import YOLO

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

prev_time = time.time()
clients = set()
server = None

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

            annotated_frame = prediction.plot().copy()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLO Hand Gesture Detection", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received. Closing application...")
                await shutdown()
                break

            await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    finally:
        clients.remove(websocket)

async def websocket_server():
    global server
    server = await websockets.serve(
        send_predictions, "0.0.0.0", 5001
    )
    await server.wait_closed()

async def shutdown():
    print("Shutting down gracefully...")

    if server:
        server.close()
        await server.wait_closed()

    for ws in clients:
        await ws.close()

    cap.release()
    cv2.destroyAllWindows()

    tasks = [t for t in asyncio.all_tasks() if not t.done()]
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    loop = asyncio.get_running_loop()
    loop.stop()

if __name__ == "__main__":
    print("WebSocket Server running on ws://localhost:5001")
    try:
        asyncio.run(websocket_server())
    except KeyboardInterrupt:
        print("Manual interrupt received.")
        asyncio.run(shutdown())
    finally:
        print("Application closed successfully.")
