from ultralytics import YOLO

model = YOLO("best.pt")

# Validate the model
metrics = model.val(confusion_matrix=True, save_dir="results/")
print(metrics.top1)
print(metrics.top5)