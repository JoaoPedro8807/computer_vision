import os
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

YOLO_API_KEY = os.environ.get('ULTRALYTICS_API_KEY')
print(f"Chave de API Ultralyics carregada: {YOLO_API_KEY}")

def train_yolo():
    """Treina o modelo YOLO v8 com os parâmetros especificados"""
    
    # Carregar modelo
    print("Carregando modelo YOLO v8m...")
    model = YOLO('yolov8m.pt')  
    
    training_params = {
        'data': 'ul://jotapeh/datasets/capsuladataset',
        'epochs': 50,  
        'batch': 16,  
        'imgsz': 640,
        'project': 'jotapeh/example-project',
        'name': 'capsula',
        'lr0': 0.02,  # Taxa inicial padrão
        'lrf': 0.001,  # Diminuir até 0.1% da taxa inicial
        'warmup_epochs': 5,
        'weight_decay': 0.0001,  
        'close_mosaic': 15,
        'patience': 50,
        'device': 0,  
        'save': True,
        'plots': True,
    }
    
    print("Iniciando treinamento com os seguintes parâmetros:")
    for param, value in training_params.items():
        print(f"  {param}: {value}")
    
    # Executar treinamento
    print("\n" + "="*60)
    print("INICIANDO TREINAMENTO")
    print("="*60 + "\n")
    
    results = model.train(**training_params)
    
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO")
    print("="*60)
    print(f"Resultados: {results}")
    
    return results


if __name__ == "__main__":
    train_yolo()
