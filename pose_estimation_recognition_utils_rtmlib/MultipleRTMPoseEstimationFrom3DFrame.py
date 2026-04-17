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
from typing import List, Union, Optional
from pose_estimation_recognition_utils import SkeletonDataPoint, SkeletonDataPointWithName, \
    SkeletonDataPointWithConfidence, SkeletonDataPointWithNameAndConfidence, PEImage, ImageSkeletonData, SkeletonGraph

from .RTMPoseEstimationFrom3DFrame import RTMPoseEstimationFrom3DFrame, add_names_to_result, add_confidence_to_result, add_names_and_confidence_to_result
from .utils import image2d_result_to_save_2d_data_with_confidence

class MultipleRTMPoseEstimationFrom3DFrame(RTMPoseEstimationFrom3DFrame):
    def __init__(self, **kwargs):
        """
        Initializes the 3D pose estimator with multi-person tracking.
        All parameters are passed through to the base class.
        """
        super().__init__(**kwargs)

    def extract_frame(self, frame: np.ndarray) -> List[List[Union[SkeletonDataPoint, SkeletonDataPointWithName,
                                                             SkeletonDataPointWithConfidence,
                                                             SkeletonDataPointWithNameAndConfidence]]]:

        """
        Extracts frames in two frames with pixel.

        Args:
            frame (np.ndarray): Video frame

        Returns:
            result: List[List[Union[SkeletonDataPoint, SkeletonDataPointWithName, SkeletonDataPointWithConfidence,
                                                             SkeletonDataPointWithNameAndConfidence]]]: 3D coordinates per person
        """

        frame_left, frame_right = self.divide_3d_frame(frame)

        #detecting the object using rtmlib
        results_left = self.model.process_image(frame_left)
        results_right = self.model.process_image(frame_right)

        pixel_list_left_persons = image2d_result_to_save_2d_data_with_confidence(results_left)
        pixel_list_right_persons = image2d_result_to_save_2d_data_with_confidence(results_right)

        all_results = []
        # Basic matching - zip for now as in parent, though tracking/matching would be better for multi-person
        for pixel_list_left, pixel_list_right in zip(pixel_list_left_persons, pixel_list_right_persons):
            result = self.sad.merge_pixel(pixel_list_left, pixel_list_right)

            if self.with_confidence:
                if self.with_names:
                    result = add_names_and_confidence_to_result(result, pixel_list_left, pixel_list_right)
                else:
                    result = add_confidence_to_result(result, pixel_list_left, pixel_list_right)
            else:
                if self.with_names:
                    result = add_names_to_result(result, self.RTMPoseNames)
            all_results.append(result)

        return all_results

    def extract_frame_to_pe(
            self,
            frame: np.ndarray,
            graph: Optional['SkeletonGraph'] = None,
            calculate_bone_vectors: bool = False
    ) -> 'PEImage':
        """
        Extracts 3D frames and returns a fully populated PEImage object.
        """
        persons_points = self.extract_frame(frame)
        pe_image = PEImage(origin=self.__class__.__name__, graph=graph)
        pe_image.HumanDetectionModel = self.model.det_model_path if hasattr(self.model, 'det_model_path') else None
        pe_image.PoseEstimationModel = self.model.pose_model_path if hasattr(self.model, 'pose_model_path') else None
        
        for p_idx, points in enumerate(persons_points):
            person_data = ImageSkeletonData(person_id=p_idx, BoundingBox=None)
            for pt in points:
                person_data.add_data_point(pt)
            pe_image.add_person(person_data)
            
        if calculate_bone_vectors and graph is not None:
            pe_image.calculate_bone_vectors()
            
        return pe_image