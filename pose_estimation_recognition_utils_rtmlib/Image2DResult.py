import numpy as np
from dataclasses import dataclass

@dataclass
class Image2DResult:
    """
    🏷️ EIN "DATEN-BEHÄLTER" FÜR EINZELBILD-ERGEBNISSE
    
    Stell dir das vor wie ein digitales Formular, das alle Infos zu 
    einer Posenerkennung in einem Bild speichert.
    
    📋 INHALT:
        frame_idx:     Bild-Nummer (bei Videos)
        keypoints:     Körperpunkt-Positionen [Personen, 133 Punkte, X/Y]
        scores:        Genauigkeiten für jeden Punkt [Personen, 133 Punkte]
        bboxes:        Begrenzungsrahmen um Personen [Personen, 5 Werte]
        num_persons:   Anzahl der gefundenen Personen
    """
    frame_idx: int
    keypoints: np.ndarray
    scores: np.ndarray
    bboxes: np.ndarray
    num_persons: int