# A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting

Collecting real dataset method and a description of what it does: 
- **IntelRealsenseViewer:** collect a bag using the Intel Realsense camera and the IntelRealsenseViewer software on Windows
- **sam2/sam2/filterpoints.py:** extract this bag into RGB and Depth images, which can extract a point cloud. Run SAM2 on the images to segment strawberries, which you can use to segment the strawberries in the pointcloud
  - For verification purposes, you can also extract the bag into RGB and Depth images, which can extract a point cloud using **IntelRealSnese**
- **sam2/sam2/transform_model_to_real.py:** to manually label the segmented point clouds (match them on top of the ground truth strawberry model) using the x, y, z position and angle scroll bars, then save the transformed model and matrix
- **PointAttn-modified/PointAttn-modified_main/test.ipynb:** to convert the saved transformations into a dataset that will be accepted as input for PointAttn 
