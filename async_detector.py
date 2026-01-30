import threading
import queue

from .hand_detector.hand_data_class import DetectHandData
from .hand_detector.hand import Hand
from .my_yolo.detection_data import ObjectDetectionData
from .my_yolo.my_yolov8 import SimpleObjectDetector
from .hand_detector.hand_detector import HandDetector
from .hand_detector.HandMovimentDetector import HandMovementDetector


class AsyncDetectorManager:
    """Gerencia inferências assíncronas de múltiplos modelos"""
    
    def __init__(self):
        # Queues para comunicação thread-safe
        self.hand_frame_queue = queue.Queue(maxsize=1)
        self.object_frame_queue = queue.Queue(maxsize=1)
        
        self.hand_result_queue = queue.Queue(maxsize=1)
        self.object_result_queue = queue.Queue(maxsize=1)
        

        #self.hand_detector = HandDetector(max_hands=2, confidence=0.5)
        self.hand_movement_detector = HandMovementDetector()
        self.object_detector = SimpleObjectDetector()
        
        self.running = True
        
        # Iniciar threads
        self.hand_thread = threading.Thread(target=self._hand_detector_worker, daemon=True)
        self.object_thread = threading.Thread(target=self._object_detector_worker, daemon=True)
        
        self.hand_thread.start()
        self.object_thread.start()
    
    def _hand_detector_worker(self):
        """ thread worker para o hand detector"""
        while self.running:
            try:
                frame = self.hand_frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                # Executar inferência
                result = self.hand_movement_detector.process_frame(frame)
                
                # Verifica se moveu de roi a para roi b
                is_moving_to_b = result.movement_a_to_b
                is_moving_to_a = result.movement_b_to_a

                #result = result.hands_detected
                
                # Colocar resultado na queue
                try:
                    self.hand_result_queue.put_nowait(result)
                except queue.Full:
                    pass  
                    
            except queue.Empty:
                continue
    
    def _object_detector_worker(self):
        """thread worker para o object detector"""
        while self.running:
            try:
                frame = self.object_frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                # Executar inferência
                result = self.object_detector.process_frame(frame)
                if result.objects is not None:
                    for obj in result.objects:
                        print(f"Objeto detectado: {obj.class_name} ({obj.class_id}) confianca: {obj.conf:.2f}")
                
                # Colocar resultado na queue
                try:
                    self.object_result_queue.put_nowait(result)
                except queue.Full:
                    pass  
                    
            except queue.Empty:
                continue
    
    def detect_hands_async(self, frame):
        """handle frame para hand detector"""
        try:
            self.hand_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def detect_objects_async(self, frame):
        """handle frame para object detector"""
        try:
            self.object_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_hand_result(self) -> DetectHandData:
        """get hand detector result """
        try:
            return self.hand_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_object_result(self) -> ObjectDetectionData:
        """get object detector result """
        try:
            return self.object_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """stop the worker threads"""
        self.running = False
        self.hand_frame_queue.put(None)
        self.object_frame_queue.put(None)
        self.hand_thread.join()
        self.object_thread.join()



