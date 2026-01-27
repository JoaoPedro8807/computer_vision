import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Tuple, Optional, List, Dict
from collections import defaultdict, deque
from .model_config import YoloConfig
from .detection_data import DetectionData, ObjectDetectionData


class SimpleObjectDetector:
    """
    Detector de objetos simplificado sem tracking e sem desenho.
    Focado em validações de confidence e contagem de detecções.
    Ideal para uso com hand_detector.
    """
    
    def __init__(self, config: YoloConfig = None, model_path: str = None):
        """
        Inicializa o detector
        
        Args:
            config: Configuração do detector (opcional)
            model_path: Caminho para o modelo YOLO customizado (opcional)
        """
        self.config = config or YoloConfig()

        model_path = model_path or self.config.MY_OBJECT_WEIGHTS
        
        if model_path:
            self.yolo_model = YOLO(model_path)
        else:
            self.yolo_model = YOLO(self.config.YOLO_MODEL)
        
        # Estado das detecções
        self.detection_history: Dict[int, DetectionData] = {}  # {class_id: DetectionData}
        self.frame_count = 0
        self.start_time = time.time()
        
    def detect_objects(self, frame: np.ndarray) -> List[DetectionData]:
        """
        Detecta objetos no frame usando YOLO
        
        Args:
            frame: Frame de vídeo (numpy array)
            
        Returns:
            Lista de objetos detectados (DetectionData)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Executar detecção YOLO
        results = self.yolo_model(
            frame,
            conf=self.config.CONFIDENCE_THRESHOLD,
            verbose=False,
            **self.config.YOLO_EXTRAS_PARAMS
        )
        
        detected_objects: List[DetectionData] = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extrair informações da detecção
                class_id = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_name = r.names[class_id]
                
                # Calcular centróide
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Verificar se já existe detecção para esta classe
                if class_id in self.detection_history:
                    # Atualizar detecção existente
                    existing_detection = self.detection_history[class_id]
                    existing_detection.update(
                        bbox=(x1, y1, x2, y2),
                        centroid=centroid,
                        conf=conf,
                        initial_time=existing_detection.first_detection_time,
                        detection_time=current_time
                    )
                    detected_objects.append(existing_detection)
                else:
                    # Criar nova detecção
                    new_detection = DetectionData(
                        bbox=(x1, y1, x2, y2),
                        centroid=centroid,
                        conf=conf,
                        class_id=class_id,
                        initial_time=current_time,
                        detection_time=current_time,
                        class_name=class_name
                    )
                    self.detection_history[class_id] = new_detection
                    detected_objects.append(new_detection)
        
        return detected_objects
    
    def get_detection_by_class(self, class_id: int) -> Optional[DetectionData]:
        """
        Retorna a detecção de uma classe específica
        
        Args:
            class_id: ID da classe a buscar
            
        Returns:
            DetectionData da classe ou None se não encontrada
        """
        return self.detection_history.get(class_id)
    
    def get_all_detections(self) -> List[DetectionData]:
        """
        Retorna todas as detecções armazenadas
        
        Returns:
            Lista de todas as DetectionData
        """
        return list(self.detection_history.values())
    
    def validate_detection(
        self,
        class_id: int,
        min_detections: int = 5,
        min_avg_confidence: float = 0.6,
        max_time: float = 10.0
    ) -> Tuple[bool, str]:
        """
        Valida se uma detecção é consistente baseado em critérios
        
        Args:
            class_id: ID da classe a validar
            min_detections: Número mínimo de detecções necessárias
            min_avg_confidence: Confiança média mínima necessária
            max_time: Tempo máximo em segundos desde a primeira detecção
            
        Returns:
            Tupla (válido: bool, mensagem: str)
        """
        detection = self.get_detection_by_class(class_id)
        
        if detection is None:
            return False, f"Classe {class_id} não detectada"
        
        # Verificar número de detecções
        if detection.detection_count < min_detections:
            return False, f"Poucas detecções: {detection.detection_count}/{min_detections}"
        
        # Calcular confiança média
        avg_confidence = self.get_average_confidence(class_id)
        if avg_confidence < min_avg_confidence:
            return False, f"Confiança baixa: {avg_confidence:.2f}/{min_avg_confidence:.2f}"
        
        # Verificar tempo
        elapsed_time = time.time() - detection.first_detection_time
        if elapsed_time > max_time:
            return False, f"Tempo excedido: {elapsed_time:.2f}s/{max_time:.2f}s"
        
        return True, f"Detecção válida: {detection.detection_count} frames, conf={avg_confidence:.2f}"
    
    def get_average_confidence(self, class_id: int) -> float:
        """
        Retorna a confiança média de uma classe
        
        Args:
            class_id: ID da classe
            
        Returns:
            Confiança média (0.0 se não encontrada)
        """
        detection = self.get_detection_by_class(class_id)
        if detection is None or len(detection.conf_history) == 0:
            return 0.0
        
        return sum(detection.conf_history) / len(detection.conf_history)
    
    def get_detection_count(self, class_id: int) -> int:
        """
        Retorna o número de vezes que uma classe foi detectada
        
        Args:
            class_id: ID da classe
            
        Returns:
            Número de detecções (0 se não encontrada)
        """
        detection = self.get_detection_by_class(class_id)
        return detection.detection_count if detection else 0
    
    def get_detection_stats(self, class_id: int) -> Optional[Dict]:
        """
        Retorna estatísticas detalhadas de uma detecção
        
        Args:
            class_id: ID da classe
            
        Returns:
            Dicionário com estatísticas ou None se não encontrada
        """
        detection = self.get_detection_by_class(class_id)
        if detection is None:
            return None
        
        conf_list = list(detection.conf_history)
        
        return {
            'class_id': class_id,
            'class_name': detection.class_name,
            'detection_count': detection.detection_count,
            'avg_confidence': sum(conf_list) / len(conf_list) if conf_list else 0.0,
            'min_confidence': min(conf_list) if conf_list else 0.0,
            'max_confidence': max(conf_list) if conf_list else 0.0,
            'current_confidence': detection.conf,
            'first_detection_time': detection.first_detection_time,
            'last_detection_time': detection.detection_time,
            'elapsed_time': time.time() - detection.first_detection_time,
            'current_bbox': detection.bbox,
            'current_centroid': detection.centroid
        }
    
    def clear_detection(self, class_id: int) -> bool:
        """
        Remove uma detecção específica do histórico
        
        Args:
            class_id: ID da classe a remover
            
        Returns:
            True se removida, False se não encontrada
        """
        if class_id in self.detection_history:
            del self.detection_history[class_id]
            return True
        return False
    
    def clear_all_detections(self):
        """Limpa todas as detecções e reseta o estado"""
        self.detection_history.clear()
        self.frame_count = 0
        self.start_time = time.time()
    
    def get_most_confident_detection(self) -> Optional[DetectionData]:
        """
        Retorna a detecção com maior confiança média
        
        Returns:
            DetectionData com maior confiança ou None se não houver detecções
        """
        if not self.detection_history:
            return None
        
        best_detection = None
        best_confidence = 0.0
        
        for detection in self.detection_history.values():
            avg_conf = self.get_average_confidence(detection.class_id)
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_detection = detection
        
        return best_detection
    
    def process_frame(self, frame: np.ndarray) -> ObjectDetectionData:
        """
        Processa um frame e retorna resultado da detecção
        
        Args:
            frame: Frame de vídeo
            
        Returns:
            ObjectDetectionData com frame e objeto detectado (mais confiante)
        """
        detected_objects = self.detect_objects(frame)
        most_confident = self.get_most_confident_detection()
        
        return ObjectDetectionData(
            frame=frame,
            object=most_confident
        )


def run_simple_detection(frame: np.ndarray, config: YoloConfig = None, model_path: str = None) -> ObjectDetectionData:
    """
    Função utilitária para detecção simples de objetos em um único frame
    
    Args:
        frame: Frame de vídeo
        config: Configuração opcional
        model_path: Caminho do modelo opcional
        
    Returns:
        ObjectDetectionData com resultado da detecção
    """
    detector = SimpleObjectDetector(config=config, model_path=model_path)
    return detector.process_frame(frame)


# Exemplo de uso
if __name__ == "__main__":
    # Teste básico com webcam
    detector = SimpleObjectDetector()
    cap = cv2.VideoCapture(0)
    
    print("Iniciando detecção simples de objetos...")
    print("Pressione 'q' para sair")
    print("Pressione 'v' para validar detecções")
    print("Pressione 'c' para limpar histórico")
    print("Pressione 's' para ver estatísticas")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Processar frame
        result = detector.process_frame(frame)
        
        # Exibir informações no console
        if result.object:
            stats = detector.get_detection_stats(result.object.class_id)
            if stats:
                print(f"\rClasse: {stats['class_name']} | "
                      f"Detecções: {stats['detection_count']} | "
                      f"Conf média: {stats['avg_confidence']:.2f} | "
                      f"Conf atual: {stats['current_confidence']:.2f}", end="")
        
        # Mostrar frame (sem desenhos)
        cv2.imshow("Simple Object Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('v'):
            # Validar todas as detecções
            print("\n\n=== VALIDAÇÃO DE DETECÇÕES ===")
            for detection in detector.get_all_detections():
                is_valid, message = detector.validate_detection(detection.class_id)
                status = "✓" if is_valid else "✗"
                print(f"{status} Classe {detection.class_name}: {message}")
        elif key == ord('c'):
            # Limpar histórico
            detector.clear_all_detections()
            print("\n\nHistórico limpo!")
        elif key == ord('s'):
            # Mostrar estatísticas
            print("\n\n=== ESTATÍSTICAS ===")
            for detection in detector.get_all_detections():
                stats = detector.get_detection_stats(detection.class_id)
                if stats:
                    print(f"\nClasse: {stats['class_name']} (ID: {stats['class_id']})")
                    print(f"  Detecções: {stats['detection_count']}")
                    print(f"  Conf média: {stats['avg_confidence']:.2f}")
                    print(f"  Conf mín/máx: {stats['min_confidence']:.2f}/{stats['max_confidence']:.2f}")
                    print(f"  Tempo decorrido: {stats['elapsed_time']:.2f}s")
    
    cap.release()
    cv2.destroyAllWindows()
