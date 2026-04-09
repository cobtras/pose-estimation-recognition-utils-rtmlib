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
RTMPoseSkeletons.py

This module provides a factory to get pre-filled SkeletonGraph objects for RTMLib models.

Author: Jonas David Stephan
Date: 2026-04-09
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Dict, List, Tuple, Optional
from pose_estimation_recognition_utils import SkeletonGraph


class RTMPoseSkeletons:
    """
    Factory class for pre-filled SkeletonGraph objects used with RTM models.
    """

    @staticmethod
    def get_skeleton_graph(model_type: int = 17) -> SkeletonGraph:
        """
        Get a SkeletonGraph for the specified model type.

        Args:
            model_type (int): Either 17 for 17-point model or 133 for 133-point model.

        Returns:
            SkeletonGraph: A pre-filled skeleton graph.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type == 17:
            return RTMPoseSkeletons.get_17_point_skeleton()
        elif model_type == 133:
            return RTMPoseSkeletons.get_133_point_skeleton()
        else:
            raise ValueError(f"Model type must be 17 or 133, got {model_type}")

    @staticmethod
    def get_17_point_skeleton() -> SkeletonGraph:
        """
        Get the standard 17-point COCO-style skeleton graph.

        Returns:
            SkeletonGraph: Graph containing edges and semantic types for the 17-point model.
        """
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),          # face
            (3, 5), (4, 6), (5, 6),                  # head to shoulders
            (5, 7), (7, 9),                          # left arm
            (6, 8), (8, 10),                         # right arm
            (5, 11), (6, 12), (11, 12),              # torso
            (11, 13), (13, 15),                      # left leg
            (12, 14), (14, 16)                       # right leg
        ]

        edge_types = {
            (0, 1): "head", (0, 2): "head", (1, 3): "head", (2, 4): "head",
            (3, 5): "head", (4, 6): "head", (5, 6): "torso",
            (5, 7): "left_arm", (7, 9): "left_arm",
            (6, 8): "right_arm", (8, 10): "right_arm",
            (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
            (11, 13): "left_leg", (13, 15): "left_leg",
            (12, 14): "right_leg", (14, 16): "right_leg"
        }

        return SkeletonGraph(edges=edges, edge_types=edge_types)

    @staticmethod
    def get_133_point_skeleton() -> SkeletonGraph:
        """
        Get the 133-point Wholebody skeleton graph.

        Note:
            Initially, only the basic 17-point body connections are included.
            Face and hand connections can be added by the user as needed.

        Returns:
            SkeletonGraph: Graph containing edges for the 133-point model.
        """
        # Basic 17-point body connections (indices 0-16 are the same)
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6), (5, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]
        
        # Add basic hip to foot connections if desired, or leave for user.
        # RTMPoseNames says: foot is 17-22.
        # For now, we stay with the basic body skeleton as the user said they will fill contents later.
        
        edge_types = {
            (0, 1): "head", (0, 2): "head", (1, 3): "head", (2, 4): "head",
            (3, 5): "head", (4, 6): "head", (5, 6): "torso",
            (5, 7): "left_arm", (7, 9): "left_arm",
            (6, 8): "right_arm", (8, 10): "right_arm",
            (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
            (11, 13): "left_leg", (13, 15): "left_leg",
            (12, 14): "right_leg", (14, 16): "right_leg"
        }

        return SkeletonGraph(edges=edges, edge_types=edge_types)
