try:
    from rtmlib import Wholebody, draw_skeleton
except ImportError:
    raise ImportError("RTMLib nicht gefunden. Installiere mit: pip install rtmlib")

from typing import Union, List, Tuple, Optional
import json
import numpy as np



class RTMPoseEstimator2D:

    def __init__(
        self,
        mode: str = 'performance',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        to_openpose: bool = False,
        kpt_threshold: float = 0.8,
        det_model_path: str = None,
        pose_model_path: str = None,
        pose_input_size: tuple = {288, 384},
        det_input_size: tuple = {640, 640}
    ):
        
        available_modes = {'performmance', 'balanced', 'lightweight', 'individual'}
        
        if mode not in available_modes:
            raise ValueError(f"Ungültiger Modus '{mode}'. Verfügbare Modi: {available_modes}")
        
        if mode == 'individual':
            if det_model_path is None or pose_model_path is None:
                raise ValueError("Für den 'individual'-Modus müssen det_model_path und pose_model_path angegeben werden.")
            if pose_input_size is None or det_input_size is None:
                raise ValueError("Für den 'individual'-Modus müssen pose_input_size und det_input_size angegeben werden.")
            
        self.backend = backend
        self.device = device
        self.to_openpose = to_openpose
        self.kpt_threshold = kpt_threshold
        self.mode = mode
        self.det_model_path = det_model_path
        self.pose_model_path = pose_model_path
        self.pose_input_size = pose_input_size
        self.det_input_size = det_input_size

        try:
            if mode == 'individual':
                self.model = Wholebody(
                    mode=mode,
                    backend=backend,
                    device=device,
                    to_openpose=to_openpose,
                    det=det_model_path,
                    pose=pose_model_path,
                    pose_input_size=pose_input_size,
                    det_input_size=det_input_size
                )
            else:
                self.model = Wholebody(
                    mode=mode,
                    backend=backend,
                    device=device,
                    to_openpose=to_openpose
                )
        except Exception as e:
            raise RuntimeError(f"Fehler beim Initialisieren des RTMLib Wholebody-Modells: {e}")