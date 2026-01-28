import cv2
import numpy as np
from collections import deque
import time
from mediapipe.python.solutions.hands import Hands
from dataclasses import dataclass
from .hand import Hand
from typing import Tuple, List


class HandDetector:
    """Detector de mãos usando MediaPipe"""
    
    def __init__(self, max_hands: int = 2, confidence: float = 0.5):
        """
        Inicializa o detector de mãos MediaPipe
        
        Args:
            max_hands: Número máximo de mãos a detectar
            confidence: Confiança mínima (0-1)
        """
        try:
            self.hands = Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=confidence,
                min_tracking_confidence=0.5
            )
            self.initialized = True
        except Exception as e:
            self.initialized = False
        
    def detect_hands(self, frame: np.ndarray) -> List[Hand]:
        """
        Detecta mãos no frame
        
        Args:
            frame: Frame de entrada (BGR)
            
        Returns:
            Lista de objetos Hand com informações das mãos detectadas
        """
        if not self.initialized:
            return []
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hands_data: List[Hand] = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            height, width, _ = frame.shape
            
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            ):
                # Extrair landmarks (21 pontos)
                landmarks = []
                xs, ys = [], []
                
                for lm in hand_landmarks.landmark:
                    x = lm.x * width
                    y = lm.y * height
                    z = lm.z
                    landmarks.append((x, y, z))
                    xs.append(x)
                    ys.append(y)
                
                # bounding box
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                # Adicionar margem ao bbox
                margin = 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)
                
                # Centroide
                centroid_x = (x1 + x2) / 2
                centroid_y = (y1 + y2) / 2
                
                confidence = handedness.classification[0].score
                hand_type = handedness.classification[0].label
                
                hand_info = Hand(
                    landmarks=landmarks,
                    bbox=(x1, y1, x2, y2),
                    handedness=hand_type,
                    confidence=confidence,
                    centroid=(centroid_x, centroid_y),
                    detection_time=time.time()
                )
                
                hands_data.append(hand_info)
        
        return hands_data
    
    def draw_hands(self, frame: np.ndarray, hands_data: List[Hand], 
                   draw_bbox: bool = True, draw_landmarks: bool = True) -> np.ndarray:
        """
        Desenha mãos detectadas no frame
        
        Args:
            frame: Frame de entrada
            hands_data: Lista de dados de mãos
            draw_bbox: Se True, desenha bounding box
            draw_landmarks: Se True, desenha landmarks (pontos)
            
        Returns:
            Frame com desenhos
        """
        frame_copy = frame.copy()
        
        for hand in hands_data:
            # Desenhar bounding box
            if draw_bbox:
                x1, y1, x2, y2 = map(int, hand.bbox)
                color = (0, 255, 0) if hand.handedness == 'Right' else (255, 0, 0)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{hand.handedness} ({hand.confidence:.2f})"
                cv2.putText(frame_copy, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Desenhar centroide
            cx, cy = map(int, hand.centroid)
            cv2.circle(frame_copy, (cx, cy), 5, (0, 0, 255), -1)
            
            # Desenhar landmarks
            if draw_landmarks:
                landmarks = hand.landmarks
                for i, (x, y, z) in enumerate(landmarks):
                    x, y = int(x), int(y)
                    cv2.circle(frame_copy, (x, y), 3, (255, 0, 0), -1)
        
        return frame_copy
    
    def get_hand_centroid(self, hand: Hand) -> Tuple[float, float]:
        """Retorna centroide da mão"""
        return hand.centroid
    
    def is_hand_moving_from_to(self, hands_history: deque, roi_a: dict, 
                               roi_b: dict, threshold: int = 10) -> bool:
        """
        Verifica se mão se moveu de ROI A para ROI B
        
        Args:
            hands_history: Deque com histórico de posições de mão
            roi_a: ROI origem
            roi_b: ROI destino
            threshold: Distância mínima entre frames para considerar movimento
            
        Returns:
            True se detectou movimento de A para B
        """
        if len(hands_history) < 5:
            return False
        
        recent_positions = list(hands_history)[-5:]
        
        first_in_a = self._is_point_in_roi(recent_positions[0], roi_a)
        
        last_in_b = self._is_point_in_roi(recent_positions[-1], roi_b)
        
        # Verificar se houve movimento contínuo
        if first_in_a and last_in_b:
            # Verificar continuidade (passou por pontos intermediários)
            rois_sequence = []
            for pos in recent_positions:
                if self._is_point_in_roi(pos, roi_a):
                    rois_sequence.append('A')
                elif self._is_point_in_roi(pos, roi_b):
                    rois_sequence.append('B')
                else:
                    rois_sequence.append('X')  # Fora das ROIs
            
            # Padrão esperado: começa em A, termina em B
            return rois_sequence[0] == 'A' and rois_sequence[-1] == 'B'
        
        return False
    
    @staticmethod
    def _is_point_in_roi(point: Tuple[float, float], roi: dict) -> bool:
        """Verifica se ponto está dentro de ROI"""
        x, y = point
        return (roi['x1'] <= x <= roi['x2'] and 
                roi['y1'] <= y <= roi['y2'])
