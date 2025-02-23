from ultralytics import YOLO
from pathlib import Path
import tqdm
import torch

def calculate_accuracy(model_path, test_dir):
    # Load the trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)

    # Initialize counters
    total = 0
    correct = 0
    confusion_matrix = {}
    class_names = list(model.names.values())

    # Initialize confusion matrix
    for true_class in class_names:
        confusion_matrix[true_class] = {pred_class: 0 for pred_class in class_names}

    # Get list of image paths and their true labels
    test_path = Path(test_dir)
    image_paths = []
    true_labels = []

    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            true_class = class_dir.name
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    true_labels.append(true_class)

    # Process predictions with progress bar
    for img_path, true_class in tqdm.tqdm(zip(image_paths, true_labels), total=len(image_paths)):
        # Run prediction
        results = model(img_path)

        # Get predicted class
        predicted_class = model.names[results[0].probs.top1]

        # Update confusion matrix
        confusion_matrix[true_class][predicted_class] += 1

        # Update counters
        total += 1
        if predicted_class == true_class:
            correct += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    # Print results
    print(f"\nTotal images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    # Print confusion matrix
    print("\nConfusion Matrix:")
    header = " " * 12 + "".join([f"{name[:8]:<8}" for name in class_names])
    print(header)

    for true_class in class_names:
        row = f"{true_class[:12]:<12}"
        for pred_class in class_names:
            count = confusion_matrix[true_class][pred_class]
            row += f"{count:<8}"
        print(row)

    return accuracy

if __name__ == "__main__":
    model_path = "../game/best.pt"  # Path to your trained model
    test_dir = "../Datasets/val"  # Path to your test directory
    calculate_accuracy(model_path, test_dir)
