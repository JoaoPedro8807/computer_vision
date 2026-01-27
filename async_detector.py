import threading
import queue
from .my_yolo.my_yolov8 import SimpleObjectDetector
from .hand_detector.hand_detector import HandDetector


class AsyncDetectorManager:
    """Gerencia inferências assíncronas de múltiplos modelos"""
    
    def __init__(self):
        # Queues para comunicação thread-safe
        self.hand_frame_queue = queue.Queue(maxsize=1)
        self.object_frame_queue = queue.Queue(maxsize=1)
        
        self.hand_result_queue = queue.Queue(maxsize=1)
        self.object_result_queue = queue.Queue(maxsize=1)
        
        self.hand_detector = HandDetector(max_hands=2, confidence=0.5)
        self.object_detector = SimpleObjectDetector()
        
        self.running = True
        
        # Iniciar threads
        self.hand_thread = threading.Thread(target=self._hand_detector_worker, daemon=True)
        self.object_thread = threading.Thread(target=self._object_detector_worker, daemon=True)
        
        self.hand_thread.start()
        self.object_thread.start()
    
    def _hand_detector_worker(self):
        """Worker thread para hand detector"""
        while self.running:
            try:
                frame = self.hand_frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                # Executar inferência
                result = self.hand_detector.detect_hands(frame)
                
                # Colocar resultado na queue
                try:
                    self.hand_result_queue.put_nowait(result)
                except queue.Full:
                    pass  # Descartar se queue está cheia
                    
            except queue.Empty:
                continue
    
    def _object_detector_worker(self):
        """Worker thread para object detector"""
        while self.running:
            try:
                frame = self.object_frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                # Executar inferência
                result = self.object_detector.process_frame(frame)
                
                # Colocar resultado na queue
                try:
                    self.object_result_queue.put_nowait(result)
                except queue.Full:
                    pass  # Descartar se queue está cheia
                    
            except queue.Empty:
                continue
    
    def detect_hands_async(self, frame):
        """Enviar frame para detecção de mão (não bloqueia)"""
        try:
            self.hand_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def detect_objects_async(self, frame):
        """Enviar frame para detecção de objeto (não bloqueia)"""
        try:
            self.object_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_hand_result(self):
        """Obter resultado de hand detector (não bloqueia)"""
        try:
            return self.hand_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_object_result(self):
        """Obter resultado de object detector (não bloqueia)"""
        try:
            return self.object_result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Para as worker threads"""
        self.running = False
        self.hand_frame_queue.put(None)
        self.object_frame_queue.put(None)
        self.hand_thread.join()
        self.object_thread.join()



