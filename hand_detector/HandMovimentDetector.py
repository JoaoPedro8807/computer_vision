import sys
from pathlib import Path


import cv2
import numpy as np
from collections import deque
from .hand_detector import HandDetector
from .hand import Hand
from .detector_config import Config
from .hand_data_class import DetectHandData
from ..my_yolo import  ObjectDetectionData, SimpleObjectDetector

class HandMovementDetector:
    
    def __init__(self, config=None):
        """Inicializa o detector de movimento de mão"""
        self.config = config or Config()
        self.hand_detector = HandDetector(max_hands=2, confidence=0.5)
        
        self.left_hand_history = deque(maxlen=30)  
        self.right_hand_history = deque(maxlen=30)
        
        self.is_triggered = False
        self.trigger_time = None
        self.movement_detected = False
        
        self.roi_a = self.config.ROI_A 
        self.roi_b = self.config.ROI_B
    
    def process_frame(self, frame: np.ndarray) -> DetectHandData:
        """
        Processa um frame e detecta movimento de mão
        """
        hands_data = self.hand_detector.detect_hands(frame)
        
        # att history
        for hand in hands_data:
            centroid = hand.centroid
            if hand.handedness == 'Left':
                self.left_hand_history.append(centroid)
            else:
                self.right_hand_history.append(centroid)
        
        # Verifica movimento
        movement_a_to_b = self.hand_detector.is_hand_moving_from_to(
            self.left_hand_history, self.roi_a, self.roi_b
        ) or self.hand_detector.is_hand_moving_from_to(
            self.right_hand_history, self.roi_a, self.roi_b
        )
        
        movement_b_to_a = self.hand_detector.is_hand_moving_from_to(
            self.left_hand_history, self.roi_b, self.roi_a
        ) or self.hand_detector.is_hand_moving_from_to(
            self.right_hand_history, self.roi_b, self.roi_a
        )
        
        frame_with_hands = self.hand_detector.draw_hands(
            frame, hands_data, 
            draw_bbox=True, 
            draw_landmarks=False  # Set para True se quiser ver os 21 pontos
        )
        
        frame_with_hands = self._draw_rois(frame_with_hands)
        
        if movement_a_to_b:
            cv2.putText(frame_with_hands, "MOVIMENTO: A -> B", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if movement_b_to_a:
            cv2.putText(frame_with_hands, "MOVIMENTO: B -> A", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return DetectHandData(
            frame=frame_with_hands,
            hands_detected=hands_data,
            movement_a_to_b=movement_a_to_b,
            movement_b_to_a=movement_b_to_a,
            left_hand_history=list(self.left_hand_history),
            right_hand_history=list(self.right_hand_history)
        )
    
    def draw_moviment_info(self, frame: np.ndarray, movement_a_to_b: bool, movement_b_to_a: bool) -> np.ndarray:
        if movement_a_to_b:
            cv2.putText(frame, "MOVIMENTO: A -> B", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if movement_b_to_a:
            cv2.putText(frame, "MOVIMENTO: B -> A", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

    def _draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Desenha ROIs no frame"""
        frame_copy = frame.copy()
        
        # ROI A
        cv2.rectangle(frame_copy, 
                     (self.roi_a['x1'], self.roi_a['y1']),
                     (self.roi_a['x2'], self.roi_a['y2']),
                     (255, 0, 0), 2)
        cv2.putText(frame_copy, "ROI A", 
                   (self.roi_a['x1'], self.roi_a['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # ROI B
        cv2.rectangle(frame_copy,
                     (self.roi_b['x1'], self.roi_b['y1']),
                     (self.roi_b['x2'], self.roi_b['y2']),
                     (0, 255, 0), 2)
        cv2.putText(frame_copy, "ROI B",
                   (self.roi_b['x1'], self.roi_b['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_copy
    
    def set_rois(self, roi_a: dict, roi_b: dict):
        """Atualiza as ROIs"""
        self.roi_a = roi_a
        self.roi_b = roi_b


def validate_object_data(object_data: ObjectDetectionData, target_object: int = 0 ) -> bool:
    """Valida se o objeto detectado é o alvo desejado"""
    if object_data is None:
        return False
    if object_data.object is None:
        return False
    if object_data.object.class_id != target_object:
        return False
    return True


def main():
    """
    Exemplo de uso com webcam em tempo real
    Pressione 'q' para sair
    """
    print("Iniciando detector de movimento de mão...")
    print("Pressione 'q' para sair\n")
    
    # Inicializar
    detector = HandMovementDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.config.FRAME_HEIGHT)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionar para melhor performance
        frame = cv2.resize(frame, (detector.config.FRAME_WIDTH, detector.config.FRAME_HEIGHT))
        
        # Processar
        result = detector.process_frame(frame)
        
        # Mostrar
        if result is not None and len(result.hands_detected) > 0:
            object_detector = SimpleObjectDetector()
            object_data = object_detector.process_frame(result.frame)
            if object_data.object is not None:
                stats = object_detector.get_detection_stats(object_data.object.class_id)
                if stats:
                    print(f"\rClasse: {stats['class_name']} | "
                        f"Detecções: {stats['detection_count']} | "
                        f"Conf média: {stats['avg_confidence']:.2f} | "
                        f"Conf atual: {stats['current_confidence']:.2f}", end="")
                        
                if validate_object_data(object_data, target_object=0):
                    print("Objeto alvo detectado com sucesso!")
                else:
                    print("Objeto alvo NÃO detectado.") 
        else:
            print("\rNenhum objeto detectado.", end="")
        
        
        #TODO exportar isso e o main para outro arquivo para centralizar o uso do my_yolo e hand_detector
        #TODO refatorar inferecnia do yolo. Está travando a img. Se possível em paralelo
        #TODO fazer acumulo de detecções para melhorar a confiabilidade e não ficar verificando de frame em frame
        #processa o objeto 


        cv2.imshow('Hand Movement Detection', result.frame)

        # Debug info
        if result.hands_detected:
            for i, hand in enumerate(result.hands_detected):
                pass
                # print(f"Mão {i} ({hand.handedness}): "
                #       f"Confiança={hand.confidence:.2f}, "
                #       f"Centroid={hand.centroid}")
        
        if result.movement_a_to_b:
            print(" MOVIMENTO DE A PARA B DETECTADO!")
        if result.movement_b_to_a:
            print(" MOVIMENTO DE B PARA A DETECTADO!")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    print("=" * 60)
    print("EXEMPLO: Detector de Movimento de Mão com MediaPipe")
    print("=" * 60)
    print("\nOpções:")
    print("1. Usar webcam (padrão)")
    print("2. Processar arquivo de vídeo")
    print("\nUse: python hand_detection_example.py [opção]")
    print("     python hand_detection_example.py webcam")
    print("     python hand_detection_example.py video seu_video.mp4")
    print("=" * 60 + "\n")
    
    main()
