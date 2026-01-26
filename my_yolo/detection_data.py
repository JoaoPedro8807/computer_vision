from collections import deque
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

class DetectionData:
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        centroid: Tuple[float, float],
        conf: float,
        class_id: int,
        initial_time: float,
        detection_time: float,
        class_name: Optional[str] = None
    ):
        self.bbox = bbox
        self.conf = conf
        self.class_id = class_id
        self.class_name = class_name
        self.centroid = centroid
        self.centroid_history: deque = deque([centroid], maxlen=100)
        self.bbox_history: deque = deque([bbox], maxlen=100)
        self.conf_history: deque = deque([conf], maxlen=100)
        self.first_detection_time = detection_time
        self.detection_time = detection_time
        self.detection_count = 1

    def update(
        self,
        bbox: Tuple[float, float, float, float],
        centroid: Tuple[float, float],
        conf: float,
        initial_time: float,
        detection_time: float
    ) -> None:
        self.bbox = bbox
        self.centroid = centroid
        self.conf = conf
        self.detection_time = detection_time
        self.initial_time = initial_time
        self.centroid_history.append(centroid)
        self.bbox_history.append(bbox)
        self.conf_history.append(conf)
        
        self.detection_count += 1

    def __str__(self) -> str:
        return (
            f"DetectionData(\n"
            f"  bbox={self.bbox},\n"
            f"  centroid={self.centroid},\n"
            f"  conf={self.conf:.2f},\n"
            f"  class_id={self.class_id},\n"
            f"  class_name={self.class_name},\n"
            f"  detection_count={self.detection_count},\n"
            f"  first_detection_time={self.first_detection_time},\n"
            f"  last_detection_time={self.detection_time}\n"
            f")"
        )
        

@dataclass
class ObjectDetectionData():
    frame: np.ndarray
    object: DetectionData





