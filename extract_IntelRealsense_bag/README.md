### Extract pointcloud, depth, and rgb information from bag file
After connecting the Intel RealSense camera to a windows computer, we can install Intel Real Sense Viewer SDK from https://www.intelrealsense.com/sdk-2/

Then record on the RealSense Viewer to retrieve a .bag file. To extract pointcloud, depth, and rgb information from bag file, run `view_bag.py` after installing pyrealsense2 matplotlib numpy. This is just to visualize the bag contents. To get the results that will be used for labeling the dataset refer to the `sam2_segmentation` directory
