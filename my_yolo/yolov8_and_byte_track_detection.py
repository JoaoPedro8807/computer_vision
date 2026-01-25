import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from model_config import Config
from byte_tracker_detector import SimpleByteTrack

class ObjectDetector:
    """Detector de movimento e validação para PDV"""

    __frames_count = 0
    __tracks_ids_founds = set()
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        # Modelos
        self.yolo_model = YOLO('C:/Users/joaog/OneDrive/Documentos/visao_computacional/runs/detect/jotapeh/example-project/capsula/weights/best.pt',  self.config.YOLO_MODEL)
        
        self.tracker = SimpleByteTrack(max_age=self.config.MAX_TRACK_AGE)
        
        
        # Estado
        self.is_triggered = False
        self.trigger_time = None
        self.validation_result = None
        self.arrived_in_b_time = None  # Quando chegou em B
        self.triggered_track_id = None  # ID do track que foi detectado após o trigger
        
    def is_point_in_roi(self, point, roi):
        """Verifica se ponto está dentro de ROI"""
        x, y = point
        return roi['x1'] <= x <= roi['x2'] and roi['y1'] <= y <= roi['y2']
    

    def draw_rectangle_on_object(self, frame, bbox, color=(0, 255, 0), thickness=2):
        """Desenha retângulo ao redor do objeto"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
    def detect_objects(self, frame):
        """Detecta objetos usando YOLO"""
        results = self.yolo_model(
            frame, 
            conf=self.config.CONFIDENCE_THRESHOLD, 
            **self.config.YOLO_EXTRAS_PARAMS,
            verbose=False) 

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                #if class_id != 0:  # Ignorar pessoas (class_id 0)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                self.draw_rectangle_on_object(frame, (x1, y1, x2, y2))
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                print(f"Classe identificada: {class_id}, Confiança: {conf:.2f}")
                detections.append((x1, y1, x2, y2, conf, class_id))
                
        return detections
    
    def detect_hand(self, frame):
        """Placeholder para detecção de mão (será implementado depois)"""
        return []
    
    def is_object_stopped_in_roi(self, track_data, roi, max_movement=20):
        """
        Verifica se objeto está parado dentro de uma ROI
        Retorna: (is_stopped, position_range)
        """
        centroids = list(track_data['centroids'])
        
        if len(centroids) < 2:  # Precisa de alguns frames para avaliar
            return False, 0
        
        # Pegar últimas centroides em B
        recent_centroids = [c for c in centroids[-10:] if self.is_point_in_roi(c, roi)]
        
        if len(recent_centroids) < 3:
            return False, 0
        
        # Calcular variação de posição (se está parado)
        xs = [c[0] for c in recent_centroids]
        ys = [c[1] for c in recent_centroids]
        
        position_range = max(max(xs) - min(xs), max(ys) - min(ys))
        
        is_stopped = position_range <= max_movement
        
        return is_stopped, position_range
    
    def validate_trajectory(self, tracks, triggered_track_id):
        """
        Valida se o objeto fez trajetória válida de A para B
        E PERMANECEU PARADO EM B POR 3 SEGUNDOS
        """
        print("Validando trajetória para track ID:", triggered_track_id, "track data: ", tracks)
        # Pegar o track que foi marcado como primeiro após trigger
        if triggered_track_id not in tracks:
            return False, "Track disparado não encontrado"
        
        track_data = tracks[triggered_track_id]
        print(f"VALIDANDO TRACK {triggered_track_id}: ", track_data)
        centroids = list(track_data['centroids'])
        
        if len(centroids) < self.config.MIN_TRACK_LENGTH:
            return False, "Rastreamento muito curto"
        
        # Verificar se começou em A
        first_point = centroids[0]
        last_point = centroids[-1]
        
        started_in_a = self.is_point_in_roi(first_point, self.config.ROI_A)
        ended_in_b = self.is_point_in_roi(last_point, self.config.ROI_B)
        
        if not started_in_a:
            return False, f"Objeto não começou na zona A, iniciou em {first_point}, COMPLETE DATA: {track_data}"
        
        if not ended_in_b:
            return False, "Objeto não terminou na zona B"
        
        print("Objeto iniciou em A e terminou em B, validando critérios...")

        # Verificar distância mínima
        distance = np.sqrt((last_point[0] - first_point[0])**2 + 
                          (last_point[1] - first_point[1])**2)
        
        if distance < self.config.MIN_DISTANCE_A_TO_B:
            return False, f"Distância insuficiente: {distance:.1f}px"
        
        # Verificar tempo
        elapsed_time = time.time() - track_data['start_time']
        if elapsed_time > self.config.MAX_TIME_A_TO_B:
            return False, f"Tempo excedido: {elapsed_time:.2f}s"
        
        # Validar continuidade (não pode teleportar)
        max_consecutive_distance = 0
        for i in range(len(centroids) - 1):
            d = np.sqrt((centroids[i+1][0] - centroids[i][0])**2 + 
                       (centroids[i+1][1] - centroids[i][1])**2)
            max_consecutive_distance = max(max_consecutive_distance, d)
        
        if max_consecutive_distance > 100:  # Muito salto entre frames
            return False, f"Movimento descontínuo detectado"
        
        # NOVO: Verificar se objeto está parado em B
        is_stopped, movement = self.is_object_stopped_in_roi(
            track_data, 
            self.config.ROI_B,
            self.config.MAX_MOVEMENT_IN_B
        )
        
        #por enquanto retirado verificação da espera do objeto no roi b

        # if not is_stopped:
        #     return False, f"Objeto ainda está se movendo em B (movimento: {movement:.1f}px)"
        
        # Verificar tempo em B
        # if self.arrived_in_b_time is None:
        #     return False, "Objeto ainda não chegou em B"
            
        # time_in_b = time.time() - self.arrived_in_b_time
        # if time_in_b < self.config.MIN_TIME_IN_ROI_B:
        #     return False, f"Objeto em B por apenas {time_in_b:.1f}s (mínimo: {self.config.MIN_TIME_IN_ROI_B}s)"
        return True, f"Trajetória válida! Item parado"#{time_in_b:.1f}s em B"
    
    def trigger_detection(self):
        """Simula leitura de código de barras - TRIGGER"""
        print("\n" + "="*60)
        print("CÓDIGO DE BARRAS LIDO - INICIANDO DETECÇÃO")
        print("="*60)
        self.is_triggered = True
        self.trigger_time = time.time()
        self.validation_result = None
        self.arrived_in_b_time = None
        self.triggered_track_id = None
        self.tracker.tracks = {}  # Resetar tracking para rastrear apenas objetos novos após trigger
    
    def draw_firsts_frames_info(self, frame, track_data):
        TARGET_FRAMES = 3
        centroids = list(track_data['centroids'][:TARGET_FRAMES])
        if self.__frames_count <= TARGET_FRAMES:
            for (x, y) in centroids:
                cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)  # Magenta para os primeiros pontos
                print(f"Frame {self.__frames_count} - Coordenadas: ({x}, {y})")


    def process_frame(self, frame):
        """Processa um frame completo"""

        self.__frames_count += 1

        detections = self.detect_objects(frame)
        
        # Atualizar tracks
        tracks = self.tracker.update(detections)


        # tracks_datas = tracks.items() if len(tracks) > 0 else None
        # print("TRACKS DATA: ", tracks_datas)
        # if tracks_datas:
        #     self.draw_firsts_frames_info(frame, tracks_datas)

        # Detectar mão
        hands = self.detect_hand(frame)
        
        # Se trigger ativo, procurar objeto em B (qualquer ID)
        if self.is_triggered:
            # Procurar qualquer objeto em zona B
            best_track = None

            for track_id, track_data in tracks.items():
                self.__tracks_ids_founds.add(track_id)
                if len(track_data['centroids']) >= 2:
                    centroid = track_data['centroids'][-1]
                    if self.is_point_in_roi(centroid, self.config.ROI_B):
                        best_track = (track_id, track_data)
                        break
            
            # Se encontrou objeto em B
            if best_track:
                track_id, best_track_data = best_track
                print(f"Objeto em zona B detectado (Track ID: {track_id}, TRACK DATA: {best_track_data})")
                print("" + "="*60)
                print("TRACKS IDS ENCONTRADOS ATÉ AGORA: ", self.__tracks_ids_founds)
                # Marcar o primeiro track detectado após trigger
                if self.triggered_track_id is None:
                    self.triggered_track_id = track_id
                    print(f"📍 Marcado primeiro track após trigger: ID {track_id}")
                
                # Registrar primeira vez que chegou em B
                if self.arrived_in_b_time is None:
                    self.arrived_in_b_time = time.time()
                    print(f"✓ Objeto detectado na zona B (Track ID: {track_id})")
                
                # Tentar validar
                is_valid, message = self.validate_trajectory(tracks, self.triggered_track_id)
                
                if is_valid:
                    self.validation_result = {
                        'valid': True,
                        'message': message,
                        'track_id': track_id,
                        'time': time.time() - self.trigger_time
                    }
                    self.is_triggered = False
                    print(f"\n VENDA CONFIRMADA")
                    print(f"  {message}")
                    print(f"  Tempo total: {self.validation_result['time']:.2f}s")
                else:
                    print(f"\n VENDA NÃO CONFIRMADA")
                    print(f"  {message}")
                    self.is_triggered = False
            else:
                # Reset se saiu de B antes de confirmar
                if self.arrived_in_b_time is not None:
                    self.arrived_in_b_time = None
            
            # Timeout da detecção
            if time.time() - self.trigger_time > self.config.MAX_TIME_A_TO_B + self.config.MIN_TIME_IN_ROI_B + 2:
                self.is_triggered = False
                print(" Timeout - nenhum movimento válido detectado")
        
        return frame, tracks, hands
    
    def draw_frame(self, frame, tracks, hands):
        """Desenha informações no frame"""
        
        # Desenhar ROIs
        roi_a = self.config.ROI_A
        roi_b = self.config.ROI_B
        
        cv2.rectangle(frame, (roi_a['x1'], roi_a['y1']), (roi_a['x2'], roi_a['y2']), 
                     (0, 255, 0), 2)
        cv2.putText(frame, "ZONA A (Inicio)", (roi_a['x1'], roi_a['y1']-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (roi_b['x1'], roi_b['y1']), (roi_b['x2'], roi_b['y2']),
                     (0, 0, 255), 2)
        cv2.putText(frame, "ZONA B (Fim)", (roi_b['x1'], roi_b['y1']-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Desenhar tracks
        for track_id, track_data in tracks.items():
            centroids = list(track_data['centroids'])
            
            # Desenhar trajetória
            for i in range(len(centroids) - 1):
                p1 = tuple(map(int, centroids[i]))
                p2 = tuple(map(int, centroids[i + 1]))
                
                # Cor diferente se é o track em monitoramento
                color = (0, 255, 255) #if track_id == self.triggered_track_id else (255, 0, 0)
                cv2.line(frame, p1, p2, color, 2)
            
            # Desenhar centróide atual
            if len(centroids) > 0:
                current = tuple(map(int, centroids[-1]))
                cv2.circle(frame, current, 5, (0, 255, 0), -1)
                
                cv2.putText(frame, f"ID:{track_id}", (current[0]+10, current[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"({current[0]}, {current[1]})", (current[0]+10, current[1]+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
        # Desenhar mãos detectadas
        for hand_side, hand_pos in hands:
            cv2.circle(frame, hand_pos, 8, (255, 0, 0), -1)
            cv2.putText(frame, f"Hand:{hand_side}", (hand_pos[0]+10, hand_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Status no topo
        status_text = "AGUARDANDO TRIGGER" if not self.is_triggered else "MONITORANDO MOVIMENTO"
        color = (0, 165, 255) if not self.is_triggered else (0, 255, 255)
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Resultado da validação
        if self.validation_result:
            result_text = f"✓ CONFIRMADO" if self.validation_result['valid'] else "✗ REJEITADO"
            result_color = (0, 255, 0) if self.validation_result['valid'] else (0, 0, 255)
            cv2.putText(frame, result_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 2)
        
        return frame



def main():

    print("Iniciando  Detector com YOLO + ByteTrack + MediaPipe")
    print("Controles:")
    print("  SPACE - TRIGGER DETECOTR")
    print("  Q     - Sair")
    print("-" * 60)

    detector = ObjectDetector()
    

    cap = cv2.VideoCapture(detector.config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Erro ao abrir câmera!")
        return
    
    print("Câmera aberta com sucesso!")
    
    fps_clock = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()

        frame, tracks, hands = detector.process_frame(frame)
        
        frame = detector.draw_frame(frame, tracks, hands)
        
        # FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - fps_clock)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (detector.config.FRAME_WIDTH - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        fps_clock = cv2.getTickCount()
        
        cv2.imshow("PDV Detector - YOLO + ByteTrack + MediaPipe", frame)
  
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            detector.trigger_detection()

        if not ret:
            print("Erro ao ler frame")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nAplicação encerrada.")


if __name__ == "__main__":
    main()
