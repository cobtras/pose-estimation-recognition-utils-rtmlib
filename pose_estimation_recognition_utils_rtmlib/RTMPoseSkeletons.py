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
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),          # face
            (3, 5), (4, 6), (5, 6),                  # head to shoulders
            (5, 7), (7, 9),                          # left arm
            (6, 8), (8, 10),                         # right arm
            (5, 11), (6, 12), (11, 12),              # torso
            (11, 13), (13, 15),                      # left leg
            (12, 14), (14, 16)                       # right leg
        ]

        edge_types = {
            (0, 1): "head", (0, 2): "head", (1, 2): "head", (1, 3): "head", (2, 4): "head",
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
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),  # face
            (3, 5), (4, 6), (5, 6),  # head to shoulders
            (5, 7), (7, 9),  # left arm
            (6, 8), (8, 10),  # right arm
            (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
            (17, 18), (17, 19), (18, 19), # left foot
            (20, 21), (20, 22), (21, 22), # right foot
            (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), #jaw
            (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), #jaw
            (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), #jaw
            (40, 41), (41, 42), (42, 43), (43, 44), # right eyebrow
            (45, 46), (46, 47), (47, 48), (48, 49), # left eyebrow
            (50, 51), (51, 52), (52, 53), (53, 54), # nose
            (54, 55), (55, 56), (56, 57), (57, 58), (53, 58), # nose
            (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 59), # right eye
            (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 65), # left eye
            (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (77, 78), # lips
            (78, 79), (79, 80), (80, 81), (81, 82), (82, 71), # lips
            (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), # mouth
            (88, 89), (89, 90), (90, 83), # mouth
            (91, 92), (92, 93), (93, 94), (94, 95), # left thump
            (96, 97), (97, 98), (98, 99), # left pinky
            (100, 101), (101, 102), (102, 103), # left middle
            (104, 105), (105, 106), (106, 107), # left ring
            (108, 109), (109, 110), (110, 111), # left little
            (112, 113), (113, 114), (114, 115), (115, 116), # right thump
            (117, 118), (118, 119), (119, 120), # right pinky
            (121, 122), (122, 123), (123, 124), # right middle
            (125, 126), (126, 127), (127, 128), # right ring
            (129, 130), (130, 131), (131, 132) # right little
        ]
        
        edge_types = {
            (0, 1): "head", (0, 2): "head", (1, 2): "head", (1, 3): "head", (2, 4): "head",
            (3, 5): "head", (4, 6): "head", (5, 6): "torso",
            (5, 7): "left_arm", (7, 9): "left_arm",
            (6, 8): "right_arm", (8, 10): "right_arm",
            (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
            (11, 13): "left_leg", (13, 15): "left_leg",
            (12, 14): "right_leg", (14, 16): "right_leg",
            (17, 18): "left_foot", (17, 19): "left_foot", (18, 19): "left_foot",
            (20, 21): "right_foot", (20, 22): "right_foot", (21, 22): "right_foot",
            (23, 24): "jaw", (24, 25): "jaw", (25, 26): "jaw", (26, 27): "jaw",
            (27, 28): "jaw", (28, 29): "jaw", (29, 30): "jaw", (30, 31): "jaw",
            (31, 32): "jaw", (32, 33): "jaw", (33, 34): "jaw", (34, 35): "jaw",
            (35, 36): "jaw", (36, 37): "jaw", (37, 38): "jaw", (38, 39): "jaw",
            (40, 41): "right_eyebrow", (41, 42): "right_eyebrow",
            (42, 43): "right_eyebrow", (43, 44): "right_eyebrow",
            (45, 46): "left_eyebrow", (46, 47): "left_eyebrow",
            (47, 48): "left_eyebrow", (48, 49): "left_eyebrow",
            (50, 51): "nose", (51, 52): "nose", (52, 53): "nose", (53, 54): "nose",
            (54, 55): "nose", (55, 56): "nose", (56, 57): "nose", (57, 58): "nose",
            (53, 58): "nose",
            (59, 60): "right_eye", (60, 61): "right_eye", (61, 62): "right_eye",
            (62, 63): "right_eye", (63, 64): "right_eye", (64, 59): "right_eye",
            (65, 66): "left_eye", (66, 67): "left_eye", (67, 68): "left_eye",
            (68, 69): "left_eye", (69, 70): "left_eye", (70, 65): "left_eye",
            (71, 72): "lips", (72, 73): "lips", (73, 74): "lips", (74, 75): "lips",
            (75, 76): "lips", (77, 78): "lips", (78, 79): "lips", (79, 80): "lips",
            (80, 81): "lips", (81, 82): "lips", (82, 71): "lips",
            (83, 84): "mouth", (84, 85): "mouth", (85, 86): "mouth", (86, 87): "mouth",
            (87, 88): "mouth", (88, 89): "mouth", (89, 90): "mouth", (90, 83): "mouth",
            (91, 92): "left_thumb", (92, 93): "left_thumb",
            (93, 94): "left_thumb", (94, 95): "left_thumb",
            (96, 97): "left_pinky", (97, 98): "left_pinky",
            (98, 99): "left_pinky",
            (100, 101): "left_middle", (101, 102): "left_middle",
            (102, 103): "left_middle",
            (104, 105): "left_ring", (105, 106): "left_ring",
            (106, 107): "left_ring",
            (108, 109): "left_little", (109, 110): "left_little",
            (110, 111): "left_little",
            (112, 113): "right_thumb", (113, 114): "right_thumb",
            (114, 115): "right_thumb", (115, 116): "right_thumb",
            (117, 118): "right_pinky", (118, 119): "right_pinky",
            (119, 120): "right_pinky",
            (121, 122): "right_middle", (122, 123): "right_middle",
            (123, 124): "right_middle",
            (125, 126): "right_ring", (126, 127): "right_ring",
            (127, 128): "right_ring",
            (129, 130): "right_little", (130, 131): "right_little",
            (131, 132): "right_little"
        }

        return SkeletonGraph(edges=edges, edge_types=edge_types)
