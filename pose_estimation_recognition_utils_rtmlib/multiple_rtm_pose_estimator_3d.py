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
Date: 2026-03-17
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from typing import Union, Optional

from .Video3DResult import Video3DResult
from .Image3DResult import Image3DResult
from .rtm_pose_estimator_3d import RTMPoseEstimator3D
from .tracking import PersonTracker

class MultipleRTMPoseEstimator3D(RTMPoseEstimator3D):
    def __init__(self, **kwargs):
        """
        Initializes the 3D pose estimator with multi-person tracking.
        All parameters are passed through to the base class.
        """
        super().__init__(**kwargs)
        self.tracker = PersonTracker()

    def process_video(
            self,
            video_path: Union[str, Path],
            max_frames: Optional[int] = None
    ) -> Video3DResult:
        """
        Verarbeitet ein Video mit stabilem Personen-Tracking.
        Die Ergebnis-Listen pro Frame haben immer die Länge der maximal jemals aufgetretenen Personen.
        Nicht sichtbare Personen werden mit Nullen aufgefüllt.
        """
        video_path=Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Video cannot be opened: {video_path}")

        fps=cap.get(cv2.CAP_PROP_FPS)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames=min(total_frames, max_frames)

        tracker=PersonTracker()
        frame_results=[]
        start_time=time.time()
        pbar=tqdm(total=total_frames, desc="Processing 3D video")

        for frame_idx in range(total_frames):
            ret, frame=cap.read()
            if not ret:
                break

            # 3D-Pose für aktuellen Frame (liefert Image3DResult)
            raw_result=self.process_image(frame, frame_idx)

            # Extrahiere 2D-Daten für Tracking
            if raw_result.num_persons > 0:
                # raw_result.bboxes_2d hat Shape (persons, 5) – wir brauchen nur [x1,y1,x2,y2] für IoU
                new_bboxes=[box[:4] for box in raw_result.bboxes_2d]
                new_keypoints=[kp for kp in raw_result.keypoints_2d]  # Liste (num_keypoints,2)
                person_ids, max_id=tracker.match_persons(new_bboxes, new_keypoints)
            else:
                person_ids=[]
                max_id=tracker.next_id - 1

            # Auffüllen auf maximale Personenanzahl (max_id+1)
            num_total=max_id + 1 if max_id >= 0 else 0
            if num_total > 0:
                # 2D-Arrays
                full_keypoints_2d=np.zeros((num_total, self.estimator.num_keypoints, 2), dtype=np.float32)
                full_scores_2d=np.zeros((num_total, self.estimator.num_keypoints), dtype=np.float32)
                full_bboxes_2d=np.zeros((num_total, 5), dtype=np.float32)

                # 3D-Arrays (keypoints_3d: persons, num_keypoints, 3; bboxes_3d: persons, 6)
                full_keypoints_3d=np.zeros((num_total, self.estimator.num_keypoints, 3), dtype=np.float32)
                full_scores_3d=np.zeros((num_total, self.estimator.num_keypoints), dtype=np.float32)
                full_bboxes_3d=np.zeros((num_total, 6), dtype=np.float32)  # laut _calculate_3d_bboxes

                if raw_result.num_persons > 0:
                    for i, pid in enumerate(person_ids):
                        full_keypoints_2d[pid]=raw_result.keypoints_2d[i]
                        full_scores_2d[pid]=raw_result.scores_2d[i]
                        full_bboxes_2d[pid]=raw_result.bboxes_2d[i]

                        full_keypoints_3d[pid]=raw_result.keypoints_3d[i]
                        full_scores_3d[pid]=raw_result.scores_3d[i]
                        full_bboxes_3d[pid]=raw_result.bboxes_3d[i]

                padded_result=Image3DResult(
                    frame_idx=raw_result.frame_idx,
                    keypoints_3d=full_keypoints_3d,
                    keypoints_2d=full_keypoints_2d,
                    scores_3d=full_scores_3d,
                    scores_2d=full_scores_2d,
                    bboxes_3d=full_bboxes_3d,
                    bboxes_2d=full_bboxes_2d,
                    num_persons=num_total,
                    method=raw_result.method
                )
            else:
                padded_result=raw_result  # leeres Result

            frame_results.append(padded_result)
            pbar.update(1)

        cap.release()
        pbar.close()
        processing_time=time.time() - start_time

        return Video3DResult(
            frame_results=frame_results,
            total_frames=len(frame_results),
            fps=fps,
            processing_time=processing_time
        )