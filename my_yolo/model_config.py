class YoloConfig:
    # Câmera
    CAMERA_ID = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    
    # ROIs (Regions of Interest)
    ROI_A = {"x1": 10, "y1": 50, "x2": 640, "y2": 670}  # Ponto A (código/início) - FRAME_WIDTH/2=640, FRAME_HEIGHT-50=670
    ROI_B = {"x1": 900, "y1": 100, "x2": 1200, "y2": 650}  # Ponto B (sacola/final)
    
    # YOLO
    YOLO_MODEL = "yolov8n.pt"
    MY_OBJECT_WEIGHTS = "C:/Users/joaog/OneDrive/Documentos/visao_computacional/runs/detect/jotapeh/example-project/capsula4/weights/best.pt"  
    YOLO_EXTRAS_PARAMS = { # Parâmetros extras para o modelo YOLO
        'batch': 6, 
        'augment': True, # Augmentar imagens aumenta precisão de inferencia e diminui tempo de inferência
        'project': 'runs/my_yolov8',
        }  
    
    CONFIDENCE_THRESHOLD = 0.5
    
    # ByteTrack
    MAX_TRACK_AGE = 60  # Frames para perder objeto (aumentado de 30)
    MIN_TRACK_LENGTH = 2  # Mínimo de frames para considerar válido
    
    # Validação de movimento
    MIN_DISTANCE_A_TO_B = 300  # Distância mínima entre A e B
    MAX_TIME_A_TO_B = 10.0  # Segundos máximos para ir de A a B
    MIN_TIME_IN_ROI_B = 10.0  # Segundos mínimos parado em B para confirmar
    MAX_MOVEMENT_IN_B = 20  # Pixels máximos de movimento em B (item "parado")
    MAX_GAP_FRAMES = 200  # Máximo de frames sem detecção para validacao

