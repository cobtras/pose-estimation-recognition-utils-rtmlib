from typing import Union, Optional
from pathlib import Path

from .Video2DResult import Video2DResult
from .rtm_pose_estimator_2d import RTMPoseEstimator2D, filter_keypoints, draw_skeleton_filtered
from .colors import PERSON_COLORS

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

    def process_image_with_annotation(self, image, draw_bbox=True, draw_keypoints=True, keypoint_threshold=0.3,
                                      ignore_keypoints=None, image_idx=0, draw_style='small'):
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

        Returns:
            Tuple of (annotated_image, Image2DResult)
        """
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
                                                           keypoint_threshold, draw_style, self.person_colors)
        return annotated, result

    def process_video(
            self,
            video_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            save_frames: bool = False,
            max_frames: Optional[int] = None
    ) -> Video2DResult:
        """
        Process 2D pose estimation on a video and return results.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save annotated frames (if save_frames is True)
            save_frames: Whether to save annotated frames
            max_frames: Maximum number of frames to process (for testing)

        Returns:
            Video2DResult containing per-frame results, total frames, fps, and processing time

        Raises:
            FileNotFoundError: If the video file does not exist
            ValueError: If the video cannot be opened
        """
        raise NotImplementedError