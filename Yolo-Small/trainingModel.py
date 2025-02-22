import torch


from ultralytics import YOLO

def main():
    # 1. Load a base classification model
    #    (e.g., yolov8n-cls.pt, yolov8s-cls.pt, etc.)
    model = YOLO('yolo11m-cls.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Train
    #    If you have a data.yaml, specify data='data.yaml'.
    #    Or directly reference your train/val directories.
    results = model.train(
        data="D:/University/Projects/Python/CSCI218-GroupAssignment/Datasets",
        epochs=10,
        imgsz=512,          # Common image size for classification
        batch=8,
        name='hand_gesture_cls',  # for logging/runs folder naming
        device=device
    )

    # 3. Print or examine results
    print("Training complete. Results:", results)

if __name__ == "__main__":
    main()