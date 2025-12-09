from dataclasses import dataclass
from typing import List
from .Image2DResult import Image2DResult

@dataclass
class Video2DResult:
    """
    🎞️ EIN "DATEN-BEHÄLTER" FÜR VIDEO-ERGEBNISSE
    
    Speichert alle Einzelbild-Ergebnisse eines Videos plus Video-Infos.
    
    📋 INHALT:
        frame_results:    Liste von PoseResult für jedes Bild
        total_frames:     Anzahl aller verarbeiteten Bilder
        fps:              Bilder pro Sekunde im Original-Video
        processing_time:  Verarbeitungszeit in Sekunden
    """
    frame_results: List[Image2DResult]
    total_frames: int
    fps: float
    processing_time: float