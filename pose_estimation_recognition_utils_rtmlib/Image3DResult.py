import numpy as np
from dataclasses import dataclass

@dataclass
class Image3DResult:
    frame_idx: int
    keypoints_3d: np.ndarray   # Shape: [Personen, 133, 3]
    keypoints_2d: np.ndarray   # Shape: [Personen, 133, 2]
    scores_3d: np.ndarray      # Shape: [Personen, 133]
    bboxes_3d: np.ndarray      # Shape: [Personen, 7] (x, y, z, w, h, d, confidence)
    num_persons: int
    method: str
    confidence: float