import numpy as np
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time

class SimpleByteTrack: #melhorar tracking
    """ByteTrack simplificado para rastreamento de objetos"""
    
    def __init__(self, max_age=30):
        self.track_id_counter = 0
        self.tracks = {}  # {track_id: Track}
        self.max_age = max_age
    
    def update(self, detections):
        """
        Atualiza tracks com novas detecções
        detections: lista de (x1, y1, x2, y2, conf, class_id)
        """
        # Calcular centróides das detecções
        detection_centroids = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id, detection_time = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detection_centroids.append({
                'centroid': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': class_id,
                'detection_time': detection_time
            })
        
        # Associar detecções com tracks existentes
        used_detections = set()
        for track_id, track in list(self.tracks.items()):
            if len(detection_centroids) == 0:
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
                continue
            
            # Encontrar detecção mais próxima
            current_pos = track['centroids'][-1]
            distances = [
                np.sqrt((det['centroid'][0] - current_pos[0])**2 + 
                        (det['centroid'][1] - current_pos[1])**2)
                for det in detection_centroids
            ]
            
            closest_idx = np.argmin(distances)
            closest_distance = distances[closest_idx]
            
            # Se distância razoável, associar
            if closest_distance < 100:  # Threshold de distância
                used_detections.add(closest_idx)
                track['centroids'].append(detection_centroids[closest_idx]['centroid'])
                track['bboxes'].append(detection_centroids[closest_idx]['bbox'])
                track['age'] = 0
            else:
                track['age'] += 1
                if track['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # Criar novos tracks para detecções não associadas
        for i, det in enumerate(detection_centroids):
            if i not in used_detections:
                self.track_id_counter += 1
                self.tracks[self.track_id_counter] = {
                    'centroids': deque([det['centroid']], maxlen=100),
                    'bboxes': deque([det['bbox']], maxlen=100),
                    'class_id': det['class_id'],
                    'conf': det['conf'],
                    'age': 0,
                    'start_time': time.time()
                }
        
        return self.tracks

