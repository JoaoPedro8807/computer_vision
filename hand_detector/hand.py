from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class Hand:
    """Dados de uma mão detectada"""
    landmarks: List[Tuple[float, float, float]]  # 21 pontos (x, y, z)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    handedness: str  # 'Left' ou 'Right'
    confidence: float  # 0-1
    centroid: Tuple[float, float]  # (x, y)
    detection_time: float  # timestamp