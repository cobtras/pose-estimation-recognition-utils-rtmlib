from typing import List

from .Image2DResult import Image2DResult
from .Image3DResult import Image3DResult
from .RTMPoseNames import RTMPoseNames
from .Video3DResult import Video3DResult

from pose_estimation_recognition_utils import (Save2DData, Save2DDataWithConfidence, Save2DDataWithName,
                                               Save2DDataWithNameAndConfidence, SkeletonDataPoint,
                                               SkeletonDataPointWithConfidence, SkeletonDataPointWithName,
                                               SkeletonDataPointWithNameAndConfidence, ImageSkeletonData,
                                               VideoSkeletonData)

def image2d_result_to_save_2d_data(result: Image2DResult) -> List[Save2DData]:

    back = []
    i = 0

    for point in result.keypoints[0]:
        back.append(Save2DData(i, point[0], point[1]))
        i+=1

    return back

def image2d_result_to_save_2d_data_with_confidence(result: Image2DResult) -> List[Save2DDataWithConfidence]:

    back = []
    i = 0

    for point in result.keypoints[0]:
        back.append(Save2DDataWithConfidence(i, point[0], point[1], result.scores[0][i]))
        i+=1

    return back

def image2d_result_to_save_2d_data_with_name(result: Image2DResult) -> List[Save2DDataWithName]:

    name_list = RTMPoseNames(model_type=result.keypoints[0].shape[0])
    back = []
    i = 0

    for point in result.keypoints[0]:
        back.append(Save2DDataWithName(i, name_list.get_name(i), point[0], point[1]))
        i+=1

    return back

def image2d_result_to_save_2d_data_with_name_and_confidence(result: Image2DResult) -> List[Save2DDataWithNameAndConfidence]:

    name_list=RTMPoseNames(model_type=result.keypoints[0].shape[0])
    back=[]
    i=0

    for point in result.keypoints[0]:
        back.append(Save2DDataWithNameAndConfidence(i, name_list.get_name(i), point[0], point[1], result.scores[0][i]))
        i+=1

    return back

def image3d_result_to_skeleton_data_point(result: Image3DResult) -> List[SkeletonDataPoint]:

    back = []
    i=0
    for point in result.keypoints_3d[0]:
        back.append(SkeletonDataPoint(i, point[0], point[1], point[2]))
        i+=1

    return back

def image3d_result_to_skeleton_data_point_with_confidence(result: Image3DResult) -> List[SkeletonDataPointWithConfidence]:

    back = []
    i=0

    for point in result.keypoints_3d[0]:
        back.append(SkeletonDataPointWithConfidence(i, point[0], point[1], point[2], result.scores_3d[0][i]))
        i+=1

    return back

def image3d_result_to_skeleton_data_point_with_name(result: Image3DResult) -> List[SkeletonDataPoint]:

    name_list=RTMPoseNames(model_type=result.keypoints_3d[0].shape[0])
    back = []
    i=0

    for point in result.keypoints_3d[0]:
        back.append(SkeletonDataPointWithName(i, name_list.get_name(i), point[0], point[1], point[2]))
        i+=1

    return back

def image3d_result_to_skeleton_data_point_with_name_and_confidence(result: Image3DResult) -> List[SkeletonDataPointWithConfidence]:

    name_list=RTMPoseNames(model_type=result.keypoints_3d[0].shape[0])
    back = []
    i=0

    for point in result.keypoints_3d[0]:
        back.append(SkeletonDataPointWithNameAndConfidence(i, name_list.get_name(i), point[0], point[1], point[2], result.scores_3d[0][i]))
        i+=1

    return back

def image3d_result_to_image_skeleton_data(result: Image3DResult) -> ImageSkeletonData:

    back = ImageSkeletonData()

    points = image3d_result_to_skeleton_data_point(result)

    for point in points:
        back.add_data_point(point)

    return back

def image3d_result_to_image_skeleton_data_with_confidence(result: Image3DResult) -> ImageSkeletonData:

    back = ImageSkeletonData()

    points = image3d_result_to_skeleton_data_point_with_confidence(result)

    for point in points:
        back.add_data_point(point)

    return back

def image3d_result_to_image_skeleton_data_with_name(result: Image3DResult) -> ImageSkeletonData:

    back = ImageSkeletonData()

    points = image3d_result_to_skeleton_data_point_with_name(result)

    for point in points:
        back.add_data_point(point)

    return back

def image3d_result_to_image_skeleton_data_with_name_and_confidence(result: Image3DResult) -> ImageSkeletonData:

    back = ImageSkeletonData()

    points = image3d_result_to_skeleton_data_point(result)

    for point in points:
        back.add_data_point(point)

    return back

def video3d_result_to_video_skeleton_data(result: Video3DResult) -> List[VideoSkeletonData]:

    back = []
    i = 0

    for frame in result.frame_results:
        vsd = VideoSkeletonData(i)

        for point in image3d_result_to_skeleton_data_point(frame):
            vsd.add_data_point(point)

        back.append(vsd)
        i+=1

    return back

def video3d_result_to_video_skeleton_data_with_confidence(result: Video3DResult) -> List[VideoSkeletonData]:

    back = []
    i = 0

    for frame in result.frame_results:
        vsd = VideoSkeletonData(i)

        for point in image3d_result_to_skeleton_data_point_with_confidence(frame):
            vsd.add_data_point(point)

        back.append(vsd)
        i+=1

    return back

def video3d_result_to_video_skeleton_data_with_name(result: Video3DResult) -> List[VideoSkeletonData]:

    back = []
    i = 0

    for frame in result.frame_results:
        vsd = VideoSkeletonData(i)

        for point in image3d_result_to_skeleton_data_point_with_name(frame):
            vsd.add_data_point(point)

        back.append(vsd)
        i+=1

    return back

def video3d_result_to_video_skeleton_data_with_name_and_confidence(result: Video3DResult) -> List[VideoSkeletonData]:

    back = []
    i = 0

    for frame in result.frame_results:
        vsd = VideoSkeletonData(i)

        for point in image3d_result_to_skeleton_data_point_with_name_and_confidence(frame):
            vsd.add_data_point(point)

        back.append(vsd)
        i+=1

    return back