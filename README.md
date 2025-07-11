# A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting

1. Collecting sim data from IsaacSim strawberry field setup
2. Collecting real dataset from an indoor strawberry field
3. Train modified PointAttn using real+sim datasets **PointAttN-Modified/PointAttN-Modified_main/models/PointAttN.py**


## Dataset
The directories data and data_test in [**PointAttN-Modified/data_sim**](https://mbzuaiac-my.sharepoint.com/personal/ali_abouzeid_mbzuai_ac_ae/_layouts/15/onedrive.aspx?e=5%3A38880696f9dd4ad9b92df73c8145c0f5&sharingv2=true&fromShare=true&at=9&CT=1739939191678&OR=OWA%2DNT%2DMail&CID=159ec6e8%2Df128%2D6d08%2D7b2e%2Da13bc2d9b57a&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNTAyMDYwNzYuMDkiLCJPUyI6IldpbmRvd3MgMTEifQ%3D%3D&cidOR=Client&FolderCTID=0x01200061DDB91AB87ED34E9DF35DE2B429C65D&id=%2Fpersonal%2Fali%5Fabouzeid%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fbags%5Fmalak%2Fdata%5Fsim%2Ezip&parent=%2Fpersonal%2Fali%5Fabouzeid%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fbags%5Fmalak) and [**PointAttN-Modified/data_mix**](https://mbzuaiac-my.sharepoint.com/personal/ali_abouzeid_mbzuai_ac_ae/_layouts/15/onedrive.aspx?e=5%3A38880696f9dd4ad9b92df73c8145c0f5&sharingv2=true&fromShare=true&at=9&CT=1739939191678&OR=OWA%2DNT%2DMail&CID=159ec6e8%2Df128%2D6d08%2D7b2e%2Da13bc2d9b57a&clickParams=eyJYLUFwcE5hbWUiOiJNaWNyb3NvZnQgT3V0bG9vayBXZWIgQXBwIiwiWC1BcHBWZXJzaW9uIjoiMjAyNTAyMDYwNzYuMDkiLCJPUyI6IldpbmRvd3MgMTEifQ%3D%3D&cidOR=Client&FolderCTID=0x01200061DDB91AB87ED34E9DF35DE2B429C65D&id=%2Fpersonal%2Fali%5Fabouzeid%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fbags%5Fmalak%2Fdata%5Fmix%2Ezip&parent=%2Fpersonal%2Fali%5Fabouzeid%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fbags%5Fmalak) contain the pure simulation data and mixed (sim+real) datasets, respectively, that were used in experimentations

1. Collecting sim data from IsaacSim strawberry field setup. Dataset link: https://app.roboflow.com/strawberry-detection-aghjb/strawberry-detection2-iilvu/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

2. Collecting real data method and a description of what each step does: 
  - **IntelRealsenseViewer:** collect a bag using the Intel Realsense camera and the IntelRealsenseViewer software on Windows
  - **sam2/sam2/filterpoints.py:** extract this bag into RGB and Depth images, which can extract a point cloud. Run SAM2 on the images to segment strawberries, which you can use to segment the strawberries in the pointcloud
    - For verification purposes, you can also extract the bag into RGB and Depth images, which can extract a point cloud using **extract_IntelRealsense_bag/view_bag.ipynb**, then view it using **extract_IntelRealsense_bag/visualize_pointcloud_files.ipynb**
  - **sam2/sam2/transform_model_to_real.py:** to manually label the segmented point clouds (match them on top of the ground truth strawberry model) using the x, y, z position and angle scroll bars, then save the transformed model and matrix
  - **PointAttN-Modified/test.ipynb:** to convert the saved transformations into a dataset that will be accepted as input for PointAttn 



# Acknowledgement
The baseline code is borrowed from [PointAttN](https://github.com/ohhhyeahhh/PointAttN)

Modification to the generation (sfa) was borrowed from [PointSea](https://github.com/czvvd/SVDFormer_PointSea)

HCD loss function was borrowed from [HyperCD](https://github.com/ark1234/ICCV2023-HyperCD)

The point clouds are visualized with [Easy3D](https://github.com/LiangliangNan/Easy3D)
