import pytest
import numpy as np
from pose_estimation_recognition_utils_rtmlib import Image2DResult, Image3DResult
from pose_estimation_recognition_utils_rtmlib.utils import (
    image2d_result_to_save_2d_data,
    image2d_result_to_save_2d_data_with_confidence,
    image2d_result_to_save_2d_data_with_name,
    image2d_result_to_save_2d_data_with_name_and_confidence,
    image3d_result_to_skeleton_data_point,
    image3d_result_to_skeleton_data_point_with_confidence,
    image3d_result_to_skeleton_data_point_with_name,
    image3d_result_to_skeleton_data_point_with_name_and_confidence,
    image2d_result_to_image_skeleton_data_2d,
    video2d_result_to_video_skeleton_data_2d,
    Video2DResult
)

@pytest.fixture
def sample_2d_result():
    keypoints = np.random.rand(1, 17, 2)
    scores = np.random.rand(1, 17)
    bboxes = np.random.rand(1, 5)
    return Image2DResult(frame_idx=0, keypoints=keypoints, scores=scores, bboxes=bboxes, num_persons=1)

@pytest.fixture
def sample_3d_result():
    keypoints_3d = np.random.rand(1, 17, 3)
    keypoints_2d = np.random.rand(1, 17, 2)
    scores_3d = np.random.rand(1, 17)
    scores_2d = np.random.rand(1, 17)
    bboxes_3d = np.random.rand(1, 7)
    bboxes_2d = np.random.rand(1, 5)
    return Image3DResult(
        frame_idx=0,
        keypoints_3d=keypoints_3d,
        keypoints_2d=keypoints_2d,
        scores_3d=scores_3d,
        scores_2d=scores_2d,
        bboxes_3d=bboxes_3d,
        bboxes_2d=bboxes_2d,
        num_persons=1,
        method='ai'
    )

@pytest.fixture
def sample_video2d_result(sample_2d_result):
    return Video2DResult(
        frame_results=[sample_2d_result, sample_2d_result],
        total_frames=2,
        fps=30.0,
        processing_time=1.0
    )

def test_image2d_result_to_save_2d_data(sample_2d_result):
    data_list = image2d_result_to_save_2d_data(sample_2d_result)
    assert len(data_list) == 17
    assert data_list[0].data['id'] == 0
    assert data_list[0].data['x'] == float(sample_2d_result.keypoints[0][0][0])

def test_image2d_result_to_save_2d_data_with_confidence(sample_2d_result):
    data_list = image2d_result_to_save_2d_data_with_confidence(sample_2d_result)
    assert len(data_list) == 17
    assert data_list[0].data['confidence'] == float(sample_2d_result.scores[0][0])

def test_image2d_result_to_save_2d_data_with_name(sample_2d_result):
    data_list = image2d_result_to_save_2d_data_with_name(sample_2d_result)
    assert len(data_list) == 17
    assert data_list[0].data['name'] == 'nose'

def test_image3d_result_to_skeleton_data_point(sample_3d_result):
    data_list = image3d_result_to_skeleton_data_point(sample_3d_result)
    assert len(data_list) == 17
    assert data_list[0].data['z'] == float(sample_3d_result.keypoints_3d[0][0][2])

def test_image3d_result_to_skeleton_data_point_with_name_and_confidence(sample_3d_result):
    # This specifically tests my fix in utils.py
    data_list = image3d_result_to_skeleton_data_point_with_name_and_confidence(sample_3d_result)
    assert len(data_list) == 17
    # Verify name exists
    assert 'name' in data_list[0].data
    assert data_list[0].data['name'] == 'nose'
    # Verify confidence exists
    assert 'confidence' in data_list[0].data
    assert data_list[0].data['confidence'] == float(sample_3d_result.scores_3d[0][0])

def test_image2d_result_to_image_skeleton_data_2d(sample_2d_result):
    skeleton_data = image2d_result_to_image_skeleton_data_2d(sample_2d_result)
    assert len(skeleton_data.get_data_points()) == 17
    assert skeleton_data.get_data_points()[0].data['x'] == float(sample_2d_result.keypoints[0][0][0])

def test_video2d_result_to_video_skeleton_data_2d(sample_video2d_result):
    video_skeleton_data_list = video2d_result_to_video_skeleton_data_2d(sample_video2d_result)
    assert len(video_skeleton_data_list) == 2
    assert video_skeleton_data_list[0].frame == 0
    assert len(video_skeleton_data_list[0].get_data_points()) == 17
