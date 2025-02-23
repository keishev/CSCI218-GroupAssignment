from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Open the video feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to visualize predictions
def visualize_prediction(frame, outputs, threshold=0.7):
    scores = outputs["logits"].softmax(-1)[0, :, :-1].max(-1).values.detach().cpu().numpy()
    labels = outputs["logits"].softmax(-1)[0, :, :-1].argmax(-1).detach().cpu().numpy()
    boxes = outputs["pred_boxes"][0].detach().cpu().numpy()

    # Convert relative box coordinates to absolute coordinates
    h, w, _ = frame.shape
    for score, box, label in zip(scores, boxes, labels):
        if score > threshold:
            # DETR outputs [cx, cy, w, h]; convert to [x_min, y_min, x_max, y_max]
            cx, cy, bw, bh = box
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)

            class_name = model.config.id2label[label]
            color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# Load the DETR model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
model.eval()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)


    # Preprocess the frame for the model
    inputs = processor(images=[frame], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Visualize predictions on the frame
    frame = visualize_prediction(frame, outputs)

    # Display the frame
    cv2.imshow("DETR Object Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
