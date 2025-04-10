# A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting

Collecting real dataset: 
- **IntelRealsenseViewer:** collect a bag using the Intel Realsense camera
- **sam2/sam2/filterpoints.py:** extract this bag into RGB and Depth images, which can extract a pointcloud. Apply SAM2 on the images to segment strawberries, then segment the strawberries in the pointcloud using segmented results from the images
- **sam2/sam2/transform_model_to_real.py:** to label the segmented pointclouds (match them on top of the ground truth strawberry model) using the x,y,z,roll,pitch,yaw scroll bars, then save the transformed model and matrix
- **PointAttn-modified/PointAttn-modified_main/test.ipynb:** to convert the saved transformations into a dataset that will be accepted as input for PointAttn 
