from urllib.parse import quote
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
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

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
    class_list = read_class_list("Objectdetection/helper.txt")
    detection_colors = generate_random_colors(class_list)

    # Replace with your camera IP, username, and password
    camera_ip = 'CCCCC'
    camera_username = 'XXXX'
    camera_password = 'CCCC'

    # URL to access the camera feed
    url = f'http://{camera_username}:{quote(camera_password)}@{camera_ip}:8080/video'

    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Cannot connect to camera at {camera_ip}")
        exit()

    # Load the YOLO model
    model = load_model("Objectdetection/yolov8n.pt", "v8")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Uncomment the line below if you want to resize the frame
        # frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Pass the model as an argument to the process_frame function
        process_frame(frame, model, class_list, detection_colors)

        cv2.imshow("ObjectDetection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
