import cv2
import torch
import time
from ultralytics import YOLO


# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a YOLO classification model
model = YOLO("game/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_time = time.time()
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run classification inference on the frame
    # You can specify imgsz=224 or another size if appropriate for your model
    results = model.predict(frame, device=device)

    # Extract classification probabilities from the first (and only) result
    prediction = results[0]
    probs = prediction.probs  # 'Probs' object
    class_names = prediction.names

    # Retrieve top-1 index and confidence
    top1_idx = probs.top1
    top1_conf = probs.top1conf  # This is a tensor/float
    top1_label = class_names[top1_idx]

    # Add the classification info (top-1) onto the frame
    text = f"{top1_label} ({float(top1_conf):.2f})"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Compute and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("YOLO Classification", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
