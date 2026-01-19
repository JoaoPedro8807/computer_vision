import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

W_WEBCAM =  640
H_WEBCAM = 480

def parse_argumatation() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YoloV8 Detector")
    parser.add_argument("--webcam-resolution", type=int, default=[W_WEBCAM, H_WEBCAM], help="Webcam index")
    return parser.parse_args()

def main():
    
    cap = cv2.VideoCapture(0)
    args = parse_argumatation()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])

    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(
    thickness=2,
    )


    print("yolo main")
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        frame = box_annotator.annotate(scene=frame, detections=detections)

        print(frame.shape)
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":

    main()

