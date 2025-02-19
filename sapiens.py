import cv2
import torch
import numpy as np
import time
from enum import Enum
from torchvision import transforms
import torch.nn.functional as F
import os

# Get the current working directory
cwd = os.getcwd()
# Build the absolute path to the model file
absolute_model_path = os.path.join(cwd, "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2")

# -----------------------------
# Define a new enum for the pose model with the absolute path
# -----------------------------
class SapiensPoseEstimationType(Enum):
    POSE_03B = absolute_model_path

# -----------------------------
# Preprocessor (similar to before)
# -----------------------------
def create_preprocessor(input_size=(1024, 768),
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])

# -----------------------------
# Sapiens Pose Estimation Model
# -----------------------------
class SapiensPoseEstimation:
    def __init__(self,
                 type: SapiensPoseEstimationType = SapiensPoseEstimationType.POSE_03B,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        # Load the pose model using the absolute path
        path = type.value
        model = torch.jit.load(path)
        model.eval()
        self.model = model.to(device).to(dtype)
        self.device = device
        self.dtype = dtype
        self.preprocessor = create_preprocessor(input_size=(1024, 768))

    def __call__(self, img: np.ndarray):
        start = time.perf_counter()
        # Convert from BGR to RGB (preprocessor expects RGB)
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input_img).to(self.device).to(self.dtype)

        with torch.inference_mode():
            # Forward pass; output is assumed to be a set of heatmaps
            heatmaps = self.model(tensor)
        keypoints = self.postprocess_pose(heatmaps, img.shape[:2])
        print(f"Pose estimation inference took: {time.perf_counter() - start:.4f} seconds")
        return keypoints

    def postprocess_pose(self, heatmaps: torch.Tensor, img_shape: tuple) -> list:
        """
        Convert the model output (heatmaps) into keypoint coordinates.
        Assumes heatmaps shape: (1, K, H, W) where K is the number of keypoints.
        """
        heatmaps = heatmaps[0].cpu().numpy()  # Shape: (K, H, W)
        num_keypoints = heatmaps.shape[0]
        H_map, W_map = heatmaps.shape[1], heatmaps.shape[2]
        keypoints = []
        # For each keypoint channel, find the (x, y) with maximum response.
        for i in range(num_keypoints):
            y_map, x_map = np.unravel_index(np.argmax(heatmaps[i]), (H_map, W_map))
            confidence = heatmaps[i, y_map, x_map]
            # If your preprocessor resizes to (1024, 768), scale the keypoints back to original size.
            orig_h, orig_w = img_shape
            scale_x = orig_w / W_map  # W_map is heatmap width
            scale_y = orig_h / H_map  # H_map is heatmap height

            keypoints.append((int(x_map * scale_x), int(y_map * scale_y), float(confidence)))
        return keypoints

# -----------------------------
# Utility function to draw keypoints on image
# -----------------------------
def draw_pose(img: np.ndarray, keypoints: list, radius=5, color=(0, 255, 0)):
    for (x, y, conf) in keypoints:
        if conf > 0.3:  # Only draw if confidence is high
            cv2.circle(img, (x, y), radius, color, -1)
    return img

import cv2
import torch
import os

# Make sure to import your pose classes and utility function:
# from your_module import SapiensPoseEstimation, SapiensPoseEstimationType, draw_pose

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Open the video feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load the Sapiens pose estimator
dtype = torch.float16
model_type = SapiensPoseEstimationType.POSE_03B  # Use the appropriate pose model variant
estimator = SapiensPoseEstimation(model_type, dtype=dtype)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame (optional, for mirror view)
    frame = cv2.flip(frame, 1)

    # Run pose estimation on the frame to get keypoints
    keypoints = estimator(frame)

    # Draw the keypoints on the frame
    frame_with_pose = draw_pose(frame, keypoints)

    # Display the result in a window
    cv2.imshow("Sapiens Pose Estimation", frame_with_pose)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
