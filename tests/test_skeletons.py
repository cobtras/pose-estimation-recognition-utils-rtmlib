import sys
import os

# Add paths to sys.path if necessary
# Assuming the current environment has the packages installed or reachable
try:
    from pose_estimation_recognition_utils_rtmlib import RTMPoseSkeletons, RTMPoseNames
    from pose_estimation_recognition_utils import SkeletonGraph
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

def test_skeletons():
    print("Testing 17-point skeleton...")
    graph_17 = RTMPoseSkeletons.get_skeleton_graph(17)
    print(f"17-point graph: {graph_17}")
    print(f"Edges: {len(graph_17.edges)}")
    assert len(graph_17.edges) == 18, f"Expected 18 edges, got {len(graph_17.edges)}"
    
    print("\nTesting 133-point skeleton...")
    graph_133 = RTMPoseSkeletons.get_skeleton_graph(133)
    print(f"133-point graph: {graph_133}")
    print(f"Edges: {len(graph_133.edges)}")
    # Currently 133 point graph has 17-point body connections
    assert len(graph_133.edges) == 18, f"Expected 18 edges, got {len(graph_133.edges)}"

    print("\nVerification successful!")

if __name__ == "__main__":
    test_skeletons()
