# Copyright 2026 Jonas David Stephan, Nathalie Dollmann
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
utils.py

This module provides utility functions for converting between different data structures.

Author: Jonas David Stephan, Nathalie Dollmann
Date: 2026-03-13
License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import List

from .Image2DResult import Image2DResult
from .Image3DResult import Image3DResult
from .RTMPoseNames import RTMPoseNames
from .Video2DResult import Video2DResult
from .Video3DResult import Video3DResult

from pose_estimation_recognition_utils import (Save2DData, Save2DDataWithConfidence, Save2DDataWithName,
                                               Save2DDataWithNameAndConfidence, SkeletonDataPoint,
                                               SkeletonDataPointWithConfidence, SkeletonDataPointWithName,
                                               SkeletonDataPointWithNameAndConfidence, ImageSkeletonData,
                                               ImageSkeletonData2D, VideoSkeletonData, VideoSkeletonData2D)

def get_2d_bbox_from_2d_result(result: Image2DResult, p_idx: int) -> List[float]:
    if hasattr(result, "bboxes") and result.bboxes is not None and len(result.bboxes) > p_idx:
        box = result.bboxes[p_idx]
        if len(box) >= 4:
            return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    return []
    
def get_bbox_from_3d_result(result: Image3DResult, p_idx: int) -> List[float]:
    if hasattr(result, "bboxes_2d") and result.bboxes_2d is not None and len(result.bboxes_2d) > p_idx:
        box = result.bboxes_2d[p_idx]
        if len(box) >= 4:
            return [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
    return []

def image2d_result_to_save_2d_data(result: Image2DResult) -> List[List[Save2DData]]:
    '''
    Function to convert Image2DResult to a list of Save2DData.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[List[Save2DData]]: A list of Save2DData objects per person.
    '''
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints[p_idx]):
            back.append(Save2DData(i, float(point[0]), float(point[1])))
        back_persons.append(back)
    return back_persons

def image2d_result_to_save_2d_data_with_confidence(result: Image2DResult) -> List[List[Save2DDataWithConfidence]]:
    '''
    Function to convert Image2DResult to a list of Save2DDataWithConfidence.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[List[Save2DDataWithConfidence]]: A list of Save2DDataWithConfidence objects per person.
    '''
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints[p_idx]):
            back.append(Save2DDataWithConfidence(i, float(point[0]), float(point[1]), float(result.scores[p_idx][i])))
        back_persons.append(back)
    return back_persons

def image2d_result_to_save_2d_data_with_name(result: Image2DResult) -> List[List[Save2DDataWithName]]:
    '''
    Function to convert Image2DResult to a list of Save2DDataWithName.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[List[Save2DDataWithName]]: A list of Save2DDataWithName objects per person.
    '''
    if result.num_persons == 0:
        return []
    name_list = RTMPoseNames(model_type=result.keypoints[0].shape[0])
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints[p_idx]):
            back.append(Save2DDataWithName(i, name_list.get_name(i), float(point[0]), float(point[1])))
        back_persons.append(back)
    return back_persons

def image2d_result_to_save_2d_data_with_name_and_confidence(result: Image2DResult) -> List[List[Save2DDataWithNameAndConfidence]]:
    '''
    Function to convert Image2DResult to a list of Save2DDataWithNameAndConfidence.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[List[Save2DDataWithNameAndConfidence]]: A list of Save2DDataWithNameAndConfidence objects per person.
    '''
    if result.num_persons == 0:
        return []
    name_list = RTMPoseNames(model_type=result.keypoints[0].shape[0])
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints[p_idx]):
            back.append(Save2DDataWithNameAndConfidence(i, name_list.get_name(i), float(point[0]), float(point[1]), result.scores[p_idx][i]))
        back_persons.append(back)
    return back_persons

def image3d_result_to_skeleton_data_point(result: Image3DResult) -> List[List[SkeletonDataPoint]]:
    '''
    Function to convert Image3DResult to a list of SkeletonDataPoint.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[List[SkeletonDataPoint]]: A list of SkeletonDataPoint objects per person.  
    '''
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints_3d[p_idx]):
            back.append(SkeletonDataPoint(i, float(point[0]), float(point[1]), float(point[2])))
        back_persons.append(back)
    return back_persons

def image3d_result_to_skeleton_data_point_with_confidence(result: Image3DResult) -> List[List[SkeletonDataPointWithConfidence]]:
    '''
    Function to convert Image3DResult to a list of SkeletonDataPointWithConfidence.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[List[SkeletonDataPointWithConfidence]]: A list of SkeletonDataPointWithConfidence objects per person. 
    '''
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints_3d[p_idx]):
            back.append(SkeletonDataPointWithConfidence(i, float(point[0]), float(point[1]), float(point[2]), float(result.scores_3d[p_idx][i])))
        back_persons.append(back)
    return back_persons

def image3d_result_to_skeleton_data_point_with_name(result: Image3DResult) -> List[List[SkeletonDataPointWithName]]:
    '''
    Function to convert Image3DResult to a list of SkeletonDataPointWithName.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[List[SkeletonDataPointWithName]]: A list of SkeletonDataPointWithName objects per person.
    '''
    if result.num_persons == 0:
        return []
    name_list = RTMPoseNames(model_type=result.keypoints_3d[0].shape[0])
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints_3d[p_idx]):
            back.append(SkeletonDataPointWithName(i, name_list.get_name(i), float(point[0]), float(point[1]), float(point[2])))
        back_persons.append(back)
    return back_persons

def image3d_result_to_skeleton_data_point_with_name_and_confidence(result: Image3DResult) -> List[List[SkeletonDataPointWithConfidence]]:
    '''
    Function to convert Image3DResult to a list of SkeletonDataPointWithNameAndConfidence.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[List[SkeletonDataPointWithNameAndConfidence]]: A list of SkeletonDataPointWithNameAndConfidence objects per person.
    '''
    if result.num_persons == 0:
        return []
    name_list = RTMPoseNames(model_type=result.keypoints_3d[0].shape[0])
    back_persons = []
    for p_idx in range(result.num_persons):
        back = []
        for i, point in enumerate(result.keypoints_3d[p_idx]):
            back.append(SkeletonDataPointWithNameAndConfidence(i, name_list.get_name(i), float(point[0]), float(point[1]), float(point[2]), result.scores_3d[p_idx][i]))
        back_persons.append(back)
    return back_persons

def image3d_result_to_image_skeleton_data(result: Image3DResult) -> List[ImageSkeletonData]:
    '''
    Function to convert Image3DResult to a list of ImageSkeletonData for multiple persons.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[ImageSkeletonData]: A list of ImageSkeletonData objects.
    '''
    persons_list = []
    points_per_person = image3d_result_to_skeleton_data_point(result)
    for p_idx in range(result.num_persons):
        bbox = get_bbox_from_3d_result(result, p_idx)
        back = ImageSkeletonData(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image3d_result_to_image_skeleton_data_with_confidence(result: Image3DResult) -> List[ImageSkeletonData]:
    '''
    Function to convert Image3DResult to a list of ImageSkeletonData.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[ImageSkeletonData]: A list of ImageSkeletonData objects.
    '''
    persons_list = []
    points_per_person = image3d_result_to_skeleton_data_point_with_confidence(result)
    for p_idx in range(result.num_persons):
        bbox = get_bbox_from_3d_result(result, p_idx)
        back = ImageSkeletonData(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image3d_result_to_image_skeleton_data_with_name(result: Image3DResult) -> List[ImageSkeletonData]:
    '''
    Function to convert Image3DResult to a list of ImageSkeletonData.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[ImageSkeletonData]: A list of ImageSkeletonData objects.
    '''
    persons_list = []
    points_per_person = image3d_result_to_skeleton_data_point_with_name(result)
    for p_idx in range(result.num_persons):
        bbox = get_bbox_from_3d_result(result, p_idx)
        back = ImageSkeletonData(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image3d_result_to_image_skeleton_data_with_name_and_confidence(result: Image3DResult) -> List[ImageSkeletonData]:
    '''
    Function to convert Image3DResult to a list of ImageSkeletonData.

    Args:
        result (Image3DResult): The input Image3DResult object.

    Returns:
        List[ImageSkeletonData]: A list of ImageSkeletonData objects.
    '''
    persons_list = []
    points_per_person = image3d_result_to_skeleton_data_point_with_name_and_confidence(result)
    for p_idx in range(result.num_persons):
        bbox = get_bbox_from_3d_result(result, p_idx)
        back = ImageSkeletonData(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def video3d_result_to_video_skeleton_data(result: Video3DResult) -> List[VideoSkeletonData]:
    '''
    Function to convert Video3DResult to a list of VideoSkeletonData.

    Args:
        result (Video3DResult): The input Video3DResult object.

    Returns:
        List[VideoSkeletonData]: A list of VideoSkeletonData objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData(i)
        persons = image3d_result_to_image_skeleton_data(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video3d_result_to_video_skeleton_data_with_confidence(result: Video3DResult) -> List[VideoSkeletonData]:
    '''
    Function to convert Video3DResult to a list of VideoSkeletonData.

    Args:
        result (Video3DResult): The input Video3DResult object.

    Returns:
        List[VideoSkeletonData]: A list of VideoSkeletonData objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData(i)
        persons = image3d_result_to_image_skeleton_data_with_confidence(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video3d_result_to_video_skeleton_data_with_name(result: Video3DResult) -> List[VideoSkeletonData]:
    '''
    Function to convert Video3DResult to a list of VideoSkeletonData.

    Args:
        result (Video3DResult): The input Video3DResult object.

    Returns:
        List[VideoSkeletonData]: A list of VideoSkeletonData objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData(i)
        persons = image3d_result_to_image_skeleton_data_with_name(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video3d_result_to_video_skeleton_data_with_name_and_confidence(result: Video3DResult) -> List[VideoSkeletonData]:
    '''
    Function to convert Video3DResult to a list of VideoSkeletonData.

    Args:
        result (Video3DResult): The input Video3DResult object.

    Returns:
        List[VideoSkeletonData]: A list of VideoSkeletonData objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData(i)
        persons = image3d_result_to_image_skeleton_data_with_name_and_confidence(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def image2d_result_to_image_skeleton_data_2d(result: Image2DResult) -> List[ImageSkeletonData2D]:
    '''
    Function to convert Image2DResult to an ImageSkeletonData2D object.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[ImageSkeletonData2D]: A list of ImageSkeletonData2D objects.
    '''
    persons_list = []
    points_per_person = image2d_result_to_save_2d_data(result)
    for p_idx in range(result.num_persons):
        bbox = get_2d_bbox_from_2d_result(result, p_idx)
        back = ImageSkeletonData2D(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image2d_result_to_image_skeleton_data_with_confidence_2d(result: Image2DResult) -> List[ImageSkeletonData2D]:
    '''
    Function to convert Image2DResult to an ImageSkeletonData2D object with confidence.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[ImageSkeletonData2D]: A list of ImageSkeletonData2D objects.
    '''
    persons_list = []
    points_per_person = image2d_result_to_save_2d_data_with_confidence(result)
    for p_idx in range(result.num_persons):
        bbox = get_2d_bbox_from_2d_result(result, p_idx)
        back = ImageSkeletonData2D(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image2d_result_to_image_skeleton_data_with_name_2d(result: Image2DResult) -> List[ImageSkeletonData2D]:
    '''
    Function to convert Image2DResult to an ImageSkeletonData2D object with names.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[ImageSkeletonData2D]: A list of ImageSkeletonData2D objects.
    '''
    persons_list = []
    points_per_person = image2d_result_to_save_2d_data_with_name(result)
    for p_idx in range(result.num_persons):
        bbox = get_2d_bbox_from_2d_result(result, p_idx)
        back = ImageSkeletonData2D(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def image2d_result_to_image_skeleton_data_with_name_and_confidence_2d(result: Image2DResult) -> List[ImageSkeletonData2D]:
    '''
    Function to convert Image2DResult to an ImageSkeletonData2D object with names and confidence.

    Args:
        result (Image2DResult): The input Image2DResult object.

    Returns:
        List[ImageSkeletonData2D]: A list of ImageSkeletonData2D objects.
    '''
    persons_list = []
    points_per_person = image2d_result_to_save_2d_data_with_name_and_confidence(result)
    for p_idx in range(result.num_persons):
        bbox = get_2d_bbox_from_2d_result(result, p_idx)
        back = ImageSkeletonData2D(person_id=p_idx, BoundingBox=bbox if bbox else None)
        for point in points_per_person[p_idx]:
            back.add_data_point(point)
        persons_list.append(back)
    return persons_list

def video2d_result_to_video_skeleton_data_2d(result: Video2DResult) -> List[VideoSkeletonData2D]:
    '''
    Function to convert Video2DResult to a list of VideoSkeletonData2D objects.

    Args:
        result (Video2DResult): The input Video2DResult object.

    Returns:
        List[VideoSkeletonData2D]: A list of VideoSkeletonData2D objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData2D(i)
        persons = image2d_result_to_image_skeleton_data_2d(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video2d_result_to_video_skeleton_data_with_confidence_2d(result: Video2DResult) -> List[VideoSkeletonData2D]:
    '''
    Function to convert Video2DResult to a list of VideoSkeletonData2D objects with confidence.

    Args:
        result (Video2DResult): The input Video2DResult object.

    Returns:
        List[VideoSkeletonData2D]: A list of VideoSkeletonData2D objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData2D(i)
        persons = image2d_result_to_image_skeleton_data_with_confidence_2d(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video2d_result_to_video_skeleton_data_with_name_2d(result: Video2DResult) -> List[VideoSkeletonData2D]:
    '''
    Function to convert Video2DResult to a list of VideoSkeletonData2D objects with names.

    Args:
        result (Video2DResult): The input Video2DResult object.

    Returns:
        List[VideoSkeletonData2D]: A list of VideoSkeletonData2D objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData2D(i)
        persons = image2d_result_to_image_skeleton_data_with_name_2d(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back

def video2d_result_to_video_skeleton_data_with_name_and_confidence_2d(result: Video2DResult) -> List[VideoSkeletonData2D]:
    '''
    Function to convert Video2DResult to a list of VideoSkeletonData2D objects with names and confidence.

    Args:
        result (Video2DResult): The input Video2DResult object.

    Returns:
        List[VideoSkeletonData2D]: A list of VideoSkeletonData2D objects.
    '''
    back = []
    for i, frame in enumerate(result.frame_results):
        vsd = VideoSkeletonData2D(i)
        persons = image2d_result_to_image_skeleton_data_with_name_and_confidence_2d(frame)
        for p in persons:
            vsd.add_person(p)
        back.append(vsd)
    return back