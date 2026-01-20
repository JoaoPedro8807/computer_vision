import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

W_WEBCAM =  640
H_WEBCAM = 480

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

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

    print("first class: ", model.model.names[0])

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()
    print("teste: ", args.webcam_resolution, type(args.webcam_resolution), tuple(args.webcam_resolution))

    zone_polygon = (ZONE_POLYGON * [args.webcam_resolution[0], args.webcam_resolution[1]]).astype(int)

    zone = sv.PolygonZone(polygon=zone_polygon)
    
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.RED, 
        text_scale=2, 
        thickness=2)

    

    print("yolo main")
    i = 0
    while True:
        
        ret, frame = cap.read()

        if not ret:
            break

        result = model(
            frame, 
            agnostic_nms=True, 
            conf=0.3, 
            iou=0.8,
            imgsz=tuple(args.webcam_resolution))[0]
        
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id != 0] #retirando a class pessoa 
        labels = [f"{model.model.names[int(class_id)]} {confidence:0.2f}" 
                  for confidence, class_id in zip(detections.confidence, detections.class_id)]
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)
        

        print(frame.shape)

        cv2.imshow("Webcam", frame)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":

    main()

