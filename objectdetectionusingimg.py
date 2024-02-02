import random
import cv2
from ultralytics import YOLO
import numpy as np

def read_class_list(file_path):
    with open(file_path, "r") as my_file:
        data = my_file.read()
        class_list = data.split("\n")
    return class_list

def generate_random_colors(class_list):
    detection_colors = []
    for _ in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        bb = random.randint(0, 255)
        detection_colors.append((b, g, r, bb))
    return detection_colors

def load_model(model_path, version):
    return YOLO(model_path, version)

def process_frame(frame, model, class_list, detection_colors):
    # Predict detections using the YOLO model
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    
    # Access the detection parameters
    DP = detect_params[0].numpy()

    # Check if there are any detections
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            # Extract information for each detection box
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box on the frame
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Add label and confidence to the detection
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

def main():
    # Read class list from helper.txt
    class_list = read_class_list("Objectdetection/helper.txt")
    
    # Generate random colors for each class
    detection_colors = generate_random_colors(class_list)
    
    # Load YOLO model
    model = load_model("Objectdetection/yolov8n.pt", "v8")

    # Read the input image
    image_path = "Objectdetection/photo.jpg"
    frame = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if frame is None:
        print("Error loading image. Make sure the image path is correct.")
        return

    # Process the single frame
    process_frame(frame, model, class_list, detection_colors)

    # Display the result and wait for a key press
    cv2.imshow("ObjectDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
