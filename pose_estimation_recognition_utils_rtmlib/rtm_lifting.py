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
rtm_lifting.py

This module provides a class to intelligently load and cache models from the Hugging Face Hub.

Author: Jonas David Stephan, Nathalie Dollmann
Date: 2025-12-18
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Optional
from model_loader import ModelLoader
from .Image2DResult import Image2DResult
from pose_estimation_recognition_utils import (
    Save2DData,
    Save2DDataWithName,
    Save2DDataWithConfidence,
    Save2DDataWithNameAndConfidence
)
from typing import List, Union

class RTMLifting:
    def __init__(self, num_keypoints: int, mode: str, local_model: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initializes the RTMLifting class.

        Args:
            num_keypoints: Number of keypoints for the pose estimation.
            mode: Mode of operation (e.g., 'train', 'eval').
            local_model: Optional local model for lifting.
            cache_dir: Optional cache directory for model storage.
        """
        self.num_keypoints = num_keypoints
        self.mode = mode

        available_modes = ['ai', 'geometric']
        if mode not in available_modes:
            raise ValueError(f"Mode '{mode}' is not supported. Choose from {available_modes}.")
        
        if mode == 'ai':
            if local_model is None:
                if num_keypoints == 17:
                    model_loader = ModelLoader(
                        repo_id="fhswf/rtm17lifting",
                        model_filename="rtm17lifting.pth",
                        cache_dir=cache_dir, 
                    )
                    self.model = model_loader.load_model(device='cpu')

                elif num_keypoints == 26:
                    raise NotImplementedError("AI lifting for 26 keypoints is not implemented yet.")
                
                elif num_keypoints == 133:
                    model_loader = ModelLoader(
                        repo_id="fhswf/rtm133lifting",
                        model_filename="rtm133lifting.pth",
                        cache_dir=cache_dir, 
                    )
                    self.model = model_loader.load_model(device='cpu')
                else:
                    raise ValueError(f"Number of keypoints '{num_keypoints}' is not supported for AI lifting.")
            else:
                model_filename = local_model.split('/')[-1]
                model_path = local_model.split('/')[:-1]
                model_loader = ModelLoader(
                    repo_id="fhswf/rtm133lifting",
                    model_filename=model_filename,
                    local_model_dir=model_path
                )
                self.model = model_loader.load_model(device='cpu')
        else:
            allowed_keypoints = [17, 26, 133]
            if num_keypoints not in allowed_keypoints:
                raise ValueError(f"Number of keypoints '{num_keypoints}' is not supported for geometric lifting.")
            if num_keypoints == 26:
                raise NotImplementedError("Geometric lifting for 26 keypoints is not implemented yet.")

    def lift_pose(self, pose_2d: Image2DResult):
        """
        Lifts a 2D pose to a 3D pose using the loaded model.

        Args:
            pose_2d: Input 2D pose data.

        Returns:
            3D pose data.
        """
        if self.mode == 'ai':
            pose_3d = self.model.lift(pose_2d)
            return pose_3d
        elif self.mode == 'geometric':
            pose_3d = self._geometric_lift(pose_2d)
            return pose_3d
        else:
            raise ValueError(f"Invalid mode '{self.mode}' for lifting.")


    def lift_pose(self, pose_2d: List[Union[Save2DData, Save2DDataWithName, Save2DDataWithConfidence, 
                                            Save2DDataWithNameAndConfidence]]):
        """
        Lifts a 2D pose to a 3D pose using the loaded model.

        Args:
            pose_2d: Input 2D pose data.

        Returns:
            3D pose data.
        """
        if self.mode == 'ai':
            pose_3d = self.model.lift(pose_2d)
            return pose_3d
        elif self.mode == 'geometric':
            pose_3d = self._geometric_lift(pose_2d)
            return pose_3d
        else:
            raise ValueError(f"Invalid mode '{self.mode}' for lifting.")