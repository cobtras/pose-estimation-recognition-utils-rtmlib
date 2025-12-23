# Copyright 2025 Jonas David Stephan, Nathalie Dollmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
rtm_pose_estimator_3d.py

This module provides utilities for 3D pose estimation using RTM models.

Author: Jonas David Stephan, Nathalie Dollmann
Date: 2025-12-23
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np
from typing import Optional, Union
from pathlib import Path
import os
import cv2
import time
from tqdm import tqdm
from .rtm_pose_estimator_2d import RTMPoseEstimator2D
from .rtm_lifting import RTMLifting
from .Image3DResult import Image3DResult
from .Video3DResult import Video3DResult

class RTMPoseEstimator3D:
    
    def __init__(
        self,
        mode: str = 'performance',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        num_keypoints: int = 133, 
        lifting_mode: str = 'ai', 
        to_openpose: bool = False,
        kpt_threshold: float = 0.8,
        det_model_path: str = None,
        pose_model_path: str = None,
        pose_input_size: tuple = (288, 384),
        det_input_size: tuple = (640, 640),
        local_model: Optional[os.PathLike] = None, 
        cache_dir: Optional[os.PathLike] = None,
    ):

        self.estimator = RTMPoseEstimator2D(
            mode=mode,
            backend=backend,
            device=device,
            to_openpose=to_openpose,
            kpt_threshold=kpt_threshold,
            det_model_path=det_model_path,
            pose_model_path=pose_model_path,
            pose_input_size=pose_input_size,
            det_input_size=det_input_size,
        )

        self.lifting = RTMLifting(
            mode=lifting_mode,
            backend=backend,
            device=device,
            num_keypoints=num_keypoints,
            local_model=local_model,
            cache_dir=cache_dir
        )

    def process_image(self, image: np.ndarray, image_idx: int = 0) -> Image3DResult:
        self.lifting.lift_pose(self.estimator.process_image(image, image_idx))

    def process_image_from_file(self, image_path: Union[str, Path]) -> Image3DResult:
        self.lifting.lift_pose(self.estimator.process_image_from_file(image_path))

    def process_video(
        self,
        video_path: Union[str, Path],
        max_frames: Optional[int] = None
    ) -> Video3DResult:
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Video cannot be opened: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frame_results = []
        start_time = time.time()
        
        pbar = tqdm(total=total_frames)
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            result_2d = self.process_image(frame, frame_idx)
            result = self.lifting.lift_pose(result_2d)
            frame_results.append(result)
       
            pbar.update(1)

        cap.release()
        pbar.close()
        
        processing_time = time.time() - start_time

        return Video3DResult(
            frame_results=frame_results,
            total_frames=len(frame_results),
            fps=fps,
            processing_time=processing_time
        )

