### Extract pointcloud, depth, and rgb information from bag file
After connecting the Intel RealSense camera to a windows computer, we can install Intel Real Sense Viewer SDK from https://www.intelrealsense.com/sdk-2/

Then record on the RealSense Viewer to retrieve a .bag file. To extract pointcloud, depth, and rgb information from bag file, run extract_bag_pcl_rgb_depth.py after installing the following libraries: !pip install pyrealsense2 matplotlib numpy
