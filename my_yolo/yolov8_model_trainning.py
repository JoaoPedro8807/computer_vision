
import os
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

YOLO_API_KEY = os.environ.get('ULTRALYTICS_API_KEY')

def train_yolo():
    """Treina o modelo YOLO v8 com os parâmetros especificados"""
    
    # Carregar modelo
    print("Carregando modelo YOLO v8m...")
    model = YOLO('yolov8m.pt')  
    
    training_params = {
        'data': 'ul://jotapeh/datasets/capsuladataset',
        'epochs': 30,
        'batch': 4,
        'imgsz': 640,
        'project': 'jotapeh/example-project',
        'name': 'capsula',
        'device': 0,  
        'patience': 20,  #early stopping
        'save': True,
        'plots': True,
        'verbose': True,
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
