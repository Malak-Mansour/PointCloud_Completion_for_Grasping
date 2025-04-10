# A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting

1. Collecting sim data from IsaacSim strawberry field setup
2. Collecting real dataset method and a description of what each step does: 
  - **IntelRealsenseViewer:** collect a bag using the Intel Realsense camera and the IntelRealsenseViewer software on Windows
  - **sam2/sam2/filterpoints.py:** extract this bag into RGB and Depth images, which can extract a point cloud. Run SAM2 on the images to segment strawberries, which you can use to segment the strawberries in the pointcloud
    - For verification purposes, you can also extract the bag into RGB and Depth images, which can extract a point cloud using **extract_IntelRealsense_bag/view_bag.ipynb**
  - **sam2/sam2/transform_model_to_real.py:** to manually label the segmented point clouds (match them on top of the ground truth strawberry model) using the x, y, z position and angle scroll bars, then save the transformed model and matrix
  - **PointAttN-Modified/PointAttN-Modified_main/test.ipynb:** to convert the saved transformations into a dataset that will be accepted as input for PointAttn 

3. Train modified PointAttn using real+sim datasets **PointAttN-Modified/PointAttN-Modified_main/models/PointAttN.py**
4. Send topics to MoveIt to test grasping
5. Deploy on robotic arm using Nvidia Jetson Orin