from .async_detector import AsyncDetectorManager
from .hand_detector.main import HandMovementDetector
from .my_yolo.my_yolov8 import SimpleObjectDetector, ObjectDetectionData
import cv2
def main():
    print("Iniciando detector async de movimento de mão...")
    
    detector = HandMovementDetector()
    async_manager = AsyncDetectorManager()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.config.FRAME_HEIGHT)
    
    hand_result = None
    object_result = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (detector.config.FRAME_WIDTH, detector.config.FRAME_HEIGHT))
        
        # Enviar frames para threads (NÃO bloqueia)
        async_manager.detect_hands_async(frame)
        
        # Obter resultados anteriores (se disponíveis)
        new_hand_result = async_manager.get_hand_result()
        if new_hand_result is not None:
            hand_result = new_hand_result
        
        new_object_result = async_manager.get_object_result()
        if new_object_result is not None:
            object_result = new_object_result
        
        # Se temos resultado de mãos, processar
        if hand_result is not None:
            # Processar e desenhar
            frame_with_hands = detector.hand_detector.draw_hands(frame, hand_result)
            frame_with_hands = detector._draw_rois(frame_with_hands)
            
            # Enviar para objeto detector (quando temos mãos)
            async_manager.detect_objects_async(frame_with_hands)
        else:
            frame_with_hands = frame
        
        # Adicionar info de objeto se disponível
        if object_result and object_result.object:
            stats = async_manager.object_detector.get_detection_stats(object_result.object.class_id)
            if stats:
                text = f"Obj: {stats['class_name']} | Conf: {stats['avg_confidence']:.2f}"
                cv2.putText(frame_with_hands, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display é rápido (não bloqueia)
        cv2.imshow('Hand Movement Detection', frame_with_hands)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    async_manager.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
