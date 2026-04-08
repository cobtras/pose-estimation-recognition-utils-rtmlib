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
rtm_pose_estimator_2d.py

This module provides a class for 2D pose estimation using RTM models for multiple persons.

Author: Jonas David Stephan
Date: 2026-03-17
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
from pathlib import Path
import time
from typing import Union, Optional, List, Tuple
from tqdm import tqdm

from .Image2DResult import Image2DResult
from .Video2DResult import Video2DResult
from .colors import PERSON_COLORS
from.tracking import PersonTracker
from .rtm_pose_estimator_2d import RTMPoseEstimator2D, filter_keypoints, draw_skeleton_filtered

import cv2

class MultipleRTMPoseEstimator2D(RTMPoseEstimator2D):

    def __init__(self, person_colors=None, **kwargs):
        """
        Initialize the RTM 2D Pose Estimator for multiple persons.

        Args:
            mode: One of 'performance', 'balanced', 'lightweight', or 'individual'
            backend: Backend to use ('onnxruntime', 'tensorrt', etc.)
            device: Device to run the model on ('cpu', 'cuda', etc.)
            to_openpose: Whether to convert keypoints to OpenPose format
            kpt_threshold: Keypoint confidence threshold for filtering
            det_model_path: Path to custom detection model (required for 'individual' mode)
            pose_model_path: Path to custom pose model (required for 'individual' mode)
            pose_input_size: Input size for the pose model (required for 'individual' mode)
            det_input_size: Input size for the detection model (required for 'individual' mode)
            num_keypoints: Number of keypoints to use

        Raises:
            ValueError: If invalid mode is provided or required parameters for 'individual' mode are missing or
                invalid keypoint number
            RuntimeError: If there is an error initializing the RTMLib Wholebody/Body model
        """
        super().__init__(**kwargs)
        self.person_colors=person_colors or PERSON_COLORS
        self._next_id=0
        self._prev_tracks=[]

    def process_image_with_annotation(self, image, draw_bbox=True, draw_keypoints=True, keypoint_threshold=0.3,
                                      ignore_keypoints=None, image_idx=0, draw_style='small', person_colors=None):
        """
        Process 2D pose estimation on an image and return annotated image.

        Args:
            image: Input image as a numpy array.
            draw_bbox: Whether to draw bounding boxes
            draw_keypoints: Whether to draw keypoints and skeleton
            keypoint_threshold: Keypoint confidence threshold for drawing
            ignore_keypoints: List of keypoint indices to ignore when drawing
            image_idx: Index of the image (for videos)
            draw_style: 'small' or 'full' for different skeleton styles
            person_colors: special colors for image

        Returns:
            Tuple of (annotated_image, Image2DResult)
        """
        person_colors=person_colors or self.person_colors
        result=self.process_image(image, image_idx)
        if ignore_keypoints is not None:
            result.keypoints, result.scores=filter_keypoints(result.keypoints, result.scores, ignore_keypoints)

        annotated=image.copy()
        if result.num_persons > 0:
            if draw_bbox:
                for i, bbox in enumerate(result.bboxes):
                    color=self.person_colors[i % len(self.person_colors)]
                    x1, y1, x2, y2=bbox[:4].astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if draw_keypoints:
                annotated=draw_skeleton_filtered(annotated, result.keypoints, result.scores, ignore_keypoints,
                                                           keypoint_threshold, draw_style, person_colors)
        return annotated, result

    def _bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Caluclates Intersection over Union für two B´bounding boxes [x1,y1,x2,y2].

        Args:
            box1: bounding box to compare with
            box2: bounding box to compare

        """
        x1=max(box1[0], box2[0])
        y1=max(box1[1], box2[1])
        x2=min(box1[2], box2[2])
        y2=min(box1[3], box2[3])
        inter=max(0, x2 - x1) * max(0, y2 - y1)
        area1=(box1[2] - box1[0]) * (box1[3] - box1[1])
        area2=(box2[2] - box2[0]) * (box2[3] - box2[1])
        union=area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _match_persons(
            self,
            new_bboxes: List[np.ndarray],
            new_keypoints: List[np.ndarray],
            old_tracks: List[dict],
            iou_threshold: float = 0.3,
            keypoint_weight: float = 0.5
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        matches persons to IDs based on IoU and keypoint similarity.

        Args:
            new_bboxes: list of bounding boxes [x1,y1,x2,y2] for actual person detections.
            new_keypoints: list of keypoint arrays (num_keypoints, 2) for actual person detections.
            old_tracks: list of Dicts with 'id', 'bbox', 'keypoints' of frame before.
            iou_threshold: threshold value for IoU (less is not a match).
            keypoint_weight: weights of keypoint usage (0 = just IoU, 1 = just keypoints).

        Returns:
            matches: liste of (new_idx, old_id)
            unmatched_new: indices of new persons without match
            unmatched_old_ids: IDs of old persons without match
        """
        n_new=len(new_bboxes)
        n_old=len(old_tracks)
        if n_old == 0:
            return [], list(range(n_new)), []

        # calculation cost matrix (less costs = better)
        cost_matrix=np.zeros((n_new, n_old))
        for i in range(n_new):
            for j in range(n_old):
                # IoU cost : 1 - IoU (higher IoU -> less costs)
                iou=self._bbox_iou(new_bboxes[i], old_tracks[j]['bbox'])
                iou_cost=1 - iou

                # Keypoint distance: mean euklidischer Abstand between valid keypoints
                kp_new=new_keypoints[i]
                kp_old=old_tracks[j]['keypoints']
                # compare only keypoints existing in both frames
                valid=np.all(kp_new != 0, axis=1) & np.all(kp_old != 0, axis=1)
                if np.sum(valid) > 0:
                    distances=np.linalg.norm(kp_new[valid] - kp_old[valid], axis=1)
                    # Normalisation:
                    kp_cost=np.mean(distances) / 100.0
                else:
                    kp_cost=1.0  # no match -> high costs

                # combined costs
                cost_matrix[i, j]=(1 - keypoint_weight) * iou_cost + keypoint_weight * kp_cost

        # Greedy Matching
        matches=[]
        matched_new=set()
        matched_old=set()

        for i in range(n_new):
            if len(matched_old) == n_old:
                break
            # looking for best old person
            # TODO treshold usage
            best_j=np.argmin(cost_matrix[i])
            if cost_matrix[i, best_j] < 1.0 and best_j not in matched_old:
                matches.append((i, old_tracks[best_j]['id']))
                matched_new.add(i)
                matched_old.add(best_j)

        unmatched_new=[i for i in range(n_new) if i not in matched_new]
        unmatched_old_ids=[old_tracks[j]['id'] for j in range(n_old) if j not in matched_old]

        return matches, unmatched_new, unmatched_old_ids

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_frames: bool = False,
        max_frames: Optional[int] = None,
        debug: bool = False,
        iou_threshold: float = 0.0,
        keypoint_weight: float = 0.5,
        bbox_smoothing: float = 0.5,
        det_threshold: float = 0.5
    ) -> Video2DResult:
        """
        Verarbeitet ein Video mit stabilem Personen-Tracking.
        Die Ergebnis-Listen pro Frame haben immer die Länge der maximal jemals aufgetretenen Personen.
        Nicht sichtbare Personen werden mit Nullen aufgefüllt.
        """
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

        if output_dir and save_frames:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        tracker = PersonTracker(iou_threshold=iou_threshold, keypoint_weight=keypoint_weight, debug=debug)
        smoothed_bboxes = {}
        frame_results = []
        start_time = time.time()
        pbar = tqdm(total=total_frames, desc="Processing video")

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 2D-Pose für aktuellen Frame
            raw_result = self.process_image(frame, frame_idx)

            if raw_result.num_persons > 0 and det_threshold > 0.0:
                valid_idx = [i for i, box in enumerate(raw_result.bboxes) if len(box) >= 5 and box[4] >= det_threshold]
                if len(valid_idx) < raw_result.num_persons:
                    if debug:
                        dropped = raw_result.num_persons - len(valid_idx)
                        dropped_scores = [box[4] for i, box in enumerate(raw_result.bboxes) if i not in valid_idx and len(box) >= 5]
                        print(f"--- Frame {frame_idx} [Filter] ---")
                        print(f"Verwerfe {dropped} Detection(s) (Scores: {[f'{s:.3f}' for s in dropped_scores]}) wegen Score < {det_threshold:.2f}")
                    
                    raw_result.bboxes = np.array([raw_result.bboxes[i] for i in valid_idx]) if len(valid_idx) > 0 else np.array([])
                    raw_result.keypoints = np.array([raw_result.keypoints[i] for i in valid_idx]) if len(valid_idx) > 0 else np.array([])
                    raw_result.scores = np.array([raw_result.scores[i] for i in valid_idx]) if len(valid_idx) > 0 else np.array([])
                    raw_result.num_persons = len(valid_idx)

            if raw_result.num_persons > 0:
                new_bboxes = [box[:4] for box in raw_result.bboxes]  # [x1,y1,x2,y2]
                new_keypoints = [kp for kp in raw_result.keypoints]  # Liste (num_keypoints,2)
                person_ids, max_id = tracker.match_persons(new_bboxes, new_keypoints)
                
                # Bounding Box Smoothing (EMA) anwenden
                if bbox_smoothing > 0.0:
                    for i, pid in enumerate(person_ids):
                        curr_box = raw_result.bboxes[i].copy()
                        if pid in smoothed_bboxes:
                            curr_box[:4] = bbox_smoothing * smoothed_bboxes[pid][:4] + (1.0 - bbox_smoothing) * curr_box[:4]
                        smoothed_bboxes[pid] = curr_box
                        raw_result.bboxes[i] = curr_box
            else:
                person_ids = []
                max_id = tracker.next_id - 1  # letzte bekannte ID

            # Auffüllen auf maximale Personenanzahl (max_id+1)
            num_total = max_id + 1 if max_id >= 0 else 0
            if num_total > 0:
                full_keypoints = np.zeros((num_total, self.num_keypoints, 2), dtype=np.float32)
                full_scores = np.zeros((num_total, self.num_keypoints), dtype=np.float32)
                full_bboxes = np.zeros((num_total, 5), dtype=np.float32)

                if raw_result.num_persons > 0:
                    for i, pid in enumerate(person_ids):
                        full_keypoints[pid] = raw_result.keypoints[i]
                        full_scores[pid] = raw_result.scores[i]
                        full_bboxes[pid] = raw_result.bboxes[i]

                padded_result = Image2DResult(
                    frame_idx=raw_result.frame_idx,
                    keypoints=full_keypoints,
                    scores=full_scores,
                    bboxes=full_bboxes,
                    num_persons=num_total
                )
            else:
                padded_result = raw_result  # leeres Result

            frame_results.append(padded_result)

            # Annotierte Frames speichern (optional)
            if save_frames and output_dir:
                # Farben nur für sichtbare Personen
                colors = None
                if raw_result.num_persons > 0:
                    colors = [self.person_colors[pid % len(self.person_colors)] for pid in person_ids]
                annotated, _ = self.process_image_with_annotation(
                    frame,
                    draw_bbox=True,
                    draw_keypoints=True,
                    keypoint_threshold=self.kpt_threshold,
                    ignore_keypoints=None,
                    image_idx=frame_idx,
                    draw_style='small',
                    person_colors=colors
                )
                frame_filename = output_dir / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_filename), annotated)

            pbar.update(1)

        cap.release()
        pbar.close()
        processing_time = time.time() - start_time

        return Video2DResult(
            frame_results=frame_results,
            total_frames=len(frame_results),
            fps=fps,
            processing_time=processing_time
        )