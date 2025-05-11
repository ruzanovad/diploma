yolo train \
    model=yolov11n.pt \
    data=datasets/experiment/dataset.yaml \
    epochs=120 \
    imgsz=500 \
    seed=42 \
    device=all \
    batch=1024 \
    verbose=False \
    dropout=0.5 \
    augment=True \
    name=yolo11pt_$(date +%Y-%m-%d_%H-%M-%S) \
    rect=True \
    plots=True \
    visualize=True \
    project=latex
