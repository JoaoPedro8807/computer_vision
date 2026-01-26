
from attr import dataclass
from typing import List
from .hand import Hand
import numpy as np

@dataclass
class DetectHandData:
    """Dados de detecção de movimento de mãos"""
    frame: np.ndarray
    hands_detected: List[Hand]
    movement_a_to_b: bool
    movement_b_to_a: bool
    left_hand_history: List[tuple]
    right_hand_history: List[tuple]