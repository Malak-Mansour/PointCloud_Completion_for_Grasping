# sim2real_pose_prediction

### Segmenting pointcloud with SAM2: 
1. From sam2: https://github.com/facebookresearch/sam2
2. conda create --name sam2 python=3.10
3. conda activate sam2
4. pip install torch==2.5.1
5. pip install torchvision==0.20.1
6. git clone https://github.com/facebookresearch/sam2.git
7. cd sam2
8. pip install -e .
9. pip install -e ".[notebooks]"
10. add the 3 files in sam2_additional_files of this repo into your sam2/sam2 folder
11. run segment_pointcloud.ipynb to segment pointcloud input using the segmented images

This is how you can view the segmented pointclouds:
'''
import os

# Directory containing the point cloud files
directory = "D:\Malak Doc\Malak Education\MBZUAI\Academic years\Spring 2025\CV703\project\segmented_pointcloud"
# good ones: 1,5,7,8=2,6,8,9 (start at 0002)

# List all files in the directory
files = os.listdir(directory)

# Visualize each point cloud
for file in files:
    file_path = os.path.join(directory, file)
    if file_path.endswith('.ply'):
        point_cloud = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([point_cloud])
'''
        

### Pointcloud stitching with TEASER to extract transformation matrix:
1. From TEASER++: https://github.com/MIT-SPARK/TEASER-plusplus
2. Follow the Minimal Python 3 example instructions
3. In examples/teaser_python_ply/teaser_python_ply.py, replace   dst_cloud = src_cloud.transform(T) with     dst_cloud = o3d.io.read_point_cloud("/home/netbot/Documents/Malak/sam2/sam2/segmented_pointcloud/segmented_0002.ply")
4. Run examples/teaser_python_ply/teaser_python_ply.py to extract the transformation matrix (estimated rotation and translation) between segmented_0001 and segmented_0002
5. Use this in the pointcloud_stitching.ipynb to stitch the image with the transformation matrix
