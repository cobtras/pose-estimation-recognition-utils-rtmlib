import pytest
from pose_estimation_recognition_utils_rtmlib import RTMPoseNames

def test_rtm_pose_names_init():
    names_17 = RTMPoseNames(model_type=17)
    assert len(names_17) == 17
    
    names_133 = RTMPoseNames(model_type=133)
    assert len(names_133) == 133

def test_rtm_pose_names_invalid():
    with pytest.raises(ValueError):
        RTMPoseNames(model_type=99)

def test_rtm_pose_names_get_name():
    names = RTMPoseNames(model_type=17)
    assert names.get_name(0) == 'nose'
    assert names[0] == 'nose'
    
    with pytest.raises(IndexError):
        names.get_name(20)

def test_rtm_pose_names_find_index():
    names = RTMPoseNames(model_type=17)
    assert names.find_index('nose') == 0
    assert names.find_index('invalid') == -1

def test_rtm_pose_names_body_part():
    names = RTMPoseNames(model_type=133)
    assert names.get_body_part(0) == 'body'
    assert names.get_body_part(17) == 'foot'
    assert names.get_body_part(23) == 'face'
    assert names.get_body_part(91) == 'left_hand'
    assert names.get_body_part(112) == 'right_hand'
