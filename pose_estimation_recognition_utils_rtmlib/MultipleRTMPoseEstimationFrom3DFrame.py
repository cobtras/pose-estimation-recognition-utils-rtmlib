# Copyright 2026 Jonas David Stephan
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

Author: Jonas David Stephan
Date: 2026-03-18
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
from typing import List, Union

from pose_estimation_recognition_utils import SkeletonDataPoint, SkeletonDataPointWithName, \
    SkeletonDataPointWithConfidence, SkeletonDataPointWithNameAndConfidence

from .RTMPoseEstimationFrom3DFrame import RTMPoseEstimationFrom3DFrame


class MultipleRTMPoseEstimationFrom3DFrame(RTMPoseEstimationFrom3DFrame):
    def __init__(self, **kwargs):
        """
        Initializes the 3D pose estimator with multi-person tracking.
        All parameters are passed through to the base class.
        """
        super().__init__(**kwargs)

    def extract_frame(self, frame: np.ndarray) -> List[Union[SkeletonDataPoint, SkeletonDataPointWithName,
                                                             SkeletonDataPointWithConfidence,
                                                             SkeletonDataPointWithNameAndConfidence]]:

        """
        Extracts frames in two frames with pixel.

        Args:
            frame (np.ndarray): Video frame

        Returns:
            result: List[Union[SkeletonDataPoint, SkeletonDataPointWithName, SkeletonDataPointWithConfidence,
                                                             SkeletonDataPointWithNameAndConfidence]]: 3D coordinates
        """

        frame_left, frame_right = self.divide_3d_frame(frame)

        #detecting the object using rtmlib
        results_left = self.model.process_image(frame_left)
        results_right = self.model.process_image(frame_right)

        #TODO