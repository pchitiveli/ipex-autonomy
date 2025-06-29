import ultralytics

model = ultralytics.YOLO("model/yolo11m-seg.pt")
model.train(data="datasets/raw_img_segmentation.yaml", imgsz=704, device=0, batch=16, epochs=500)