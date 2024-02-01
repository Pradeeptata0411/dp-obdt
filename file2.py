from ultralytics import YOLO

# load a pretrained YOLOv8n model
model = YOLO("Objectdetection/yolov8n.pt", "v8")
# predict on an image
detection_output = model.predict(source="Objectdetection/img1.jpeg", conf=0.25, save=True)

# # Display tensor array
# print(detection_output)

# # Display numpy array
# print(detection_output[0].numpy())